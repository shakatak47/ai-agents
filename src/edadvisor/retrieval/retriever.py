from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from loguru import logger

from edadvisor.config import settings


@dataclass
class RetrievalResult:
    docs: list[Document]
    scores: list[float]
    confidence: float
    should_escalate: bool
    query: str


def _confidence_from_scores(scores: list[float], threshold: float) -> float:
    """
    Simple confidence score: weighted average of similarity scores,
    penalised by the fraction of results below the threshold.
    Range: 0.0 (no useful results) → 1.0 (all results very relevant).
    """
    if not scores:
        return 0.0

    above = [s for s in scores if s >= threshold]
    mean_score = float(np.mean(scores))
    coverage = len(above) / len(scores)

    return round(mean_score * coverage, 4)


def retrieve(
    query: str,
    store: FAISS,
    top_k_initial: int | None = None,
    top_k_final: int | None = None,
    threshold: float | None = None,
    metadata_filter: dict | None = None,
) -> RetrievalResult:
    """
    Two-stage retrieval:
      1. Fetch top_k_initial candidates via FAISS similarity search
      2. Re-rank with MMR (Maximal Marginal Relevance) to get top_k_final
         diverse, relevant results

    MMR trades off relevance vs diversity — avoids returning 5 copies
    of the same paragraph about a course deadline.
    """
    k0 = top_k_initial or settings.top_k_retrieval
    kf = top_k_final   or settings.top_k_final
    thr = threshold    or settings.similarity_threshold

    # similarity search with scores
    try:
        raw = store.similarity_search_with_score(query, k=k0)
    except Exception as e:
        logger.error(f"FAISS search failed: {e}")
        return RetrievalResult(docs=[], scores=[], confidence=0.0, should_escalate=True, query=query)

    # filter by threshold
    filtered = [(doc, score) for doc, score in raw if score >= thr]
    if not filtered:
        logger.warning(f"no results above threshold={thr}  query='{query[:60]}'")
        return RetrievalResult(
            docs=[], scores=[], confidence=0.0, should_escalate=True, query=query
        )

    # MMR re-ranking for diversity
    try:
        mmr_docs = store.max_marginal_relevance_search(query, k=kf, fetch_k=k0)
    except Exception:
        mmr_docs = [doc for doc, _ in filtered[:kf]]

    scores = [score for _, score in filtered[:kf]]
    confidence = _confidence_from_scores(scores, thr)
    should_escalate = confidence < settings.confidence_escalation_threshold

    if should_escalate:
        logger.info(f"low confidence={confidence:.3f}  escalating  query='{query[:60]}'")
    else:
        logger.debug(f"retrieved {len(mmr_docs)} docs  confidence={confidence:.3f}")

    return RetrievalResult(
        docs=mmr_docs,
        scores=scores,
        confidence=confidence,
        should_escalate=should_escalate,
        query=query,
    )
