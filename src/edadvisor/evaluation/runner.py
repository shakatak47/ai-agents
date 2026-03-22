"""
Evaluation pipeline — measures answer quality using ragas metrics.

Metrics:
  faithfulness      — is the answer grounded in the retrieved context?
  answer_relevancy  — does the answer actually address the question?
  context_precision — are the retrieved docs relevant to the question?
  context_recall    — did we retrieve all the context needed?

Usage:
    python -m edadvisor.evaluation.runner
    python -m edadvisor.evaluation.runner --test-set evaluation/test_set.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


TEST_SET_PATH = Path("evaluation/test_set.json")
RESULTS_DIR   = Path("evaluation/results")

# CI gates — build fails if these aren't met
GATE = {
    "faithfulness":     0.75,
    "answer_relevancy": 0.70,
}


def load_test_set(path: Path = TEST_SET_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Test set not found: {path}\n"
            "See evaluation/test_set.json for the required format."
        )
    data = json.loads(path.read_text())
    logger.info(f"loaded {len(data)} test cases from {path}")
    return data


def run_queries(test_cases: list[dict]) -> list[dict]:
    """Send each test question through the RAG chain and collect answers."""
    from edadvisor.config import settings
    from edadvisor.generation.chain import RAGChain
    from edadvisor.ingestion.store import get_embedder, load_index

    embedder = get_embedder()
    store    = load_index(settings.vector_store_path, embedder)
    chain    = RAGChain(store)

    results = []
    for i, case in enumerate(test_cases):
        session = f"eval-{i:04d}"
        resp = chain.query(case["question"], session_id=session)
        results.append({
            "question":         case["question"],
            "ground_truth":     case["ground_truth"],
            "answer":           resp.answer,
            "contexts":         [d.page_content for d in (resp.retrieval.docs if resp.retrieval else [])],
            "category":         case.get("category", ""),
            "escalated":        resp.escalated,
            "confidence":       resp.confidence,
        })
        logger.debug(f"[{i+1}/{len(test_cases)}] done")

    return results


def compute_ragas(results: list[dict]) -> dict:
    """Run ragas metrics over the collected results."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        ds = Dataset.from_list([
            {
                "question":  r["question"],
                "answer":    r["answer"],
                "contexts":  r["contexts"] or [""],
                "ground_truth": r["ground_truth"],
            }
            for r in results
        ])
        scores = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
        return dict(scores)
    except ImportError:
        logger.warning("ragas not installed — skipping metric computation")
        return {}


def save_results(results: list[dict], scores: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = RESULTS_DIR / f"eval_{ts}.json"
    out.write_text(json.dumps({"scores": scores, "cases": results}, indent=2))
    logger.info(f"results saved → {out}")
    return out


def check_gates(scores: dict) -> bool:
    """Returns True if all gate thresholds are met."""
    passed = True
    for metric, threshold in GATE.items():
        actual = scores.get(metric, 0.0)
        ok = actual >= threshold
        status = "✓" if ok else "✗ FAIL"
        logger.info(f"  {status}  {metric}: {actual:.3f}  (gate={threshold})")
        if not ok:
            passed = False
    return passed


def main(test_set_path: str | None = None) -> None:
    path = Path(test_set_path) if test_set_path else TEST_SET_PATH
    logger.info("=== EdAdvisor evaluation ===")

    test_cases = load_test_set(path)
    results    = run_queries(test_cases)
    scores     = compute_ragas(results)
    out_path   = save_results(results, scores)

    logger.info("--- scores ---")
    for k, v in scores.items():
        logger.info(f"  {k}: {v:.4f}")

    passed = check_gates(scores)
    if not passed:
        logger.error("evaluation gates FAILED — review results and improve prompt/chunking")
        sys.exit(1)

    logger.info(f"evaluation PASSED  report={out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--test-set", default=None)
    args = p.parse_args()
    main(args.test_set)
