from __future__ import annotations

from pathlib import Path

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from loguru import logger
from sentence_transformers import SentenceTransformer

from edadvisor.config import settings


class EmbedderWrapper:
    """
    Thin wrapper so SentenceTransformer works as a LangChain embedding class.
    LangChain's built-in HuggingFaceEmbeddings has extra deps we don't need.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        logger.info(f"loading embedding model: {model_name}")
        self._model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self._model.encode([text], normalize_embeddings=True)
        return vec[0].tolist()


def build_index(chunks: list[Document], embedder: EmbedderWrapper) -> FAISS:
    """
    Build a FAISS index from a list of Document chunks.
    Uses IndexFlatIP (inner product on normalised vectors = cosine similarity).
    """
    if not chunks:
        raise ValueError("no chunks to index")

    logger.info(f"building FAISS index from {len(chunks):,} chunks…")
    store = FAISS.from_documents(chunks, embedder)
    logger.info("index built")
    return store


def save_index(store: FAISS, path: str | Path) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    store.save_local(str(p))
    logger.info(f"index saved → {p}")


def load_index(path: str | Path, embedder: EmbedderWrapper) -> FAISS:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"FAISS index not found: {p}\nRun `make ingest` first.")
    store = FAISS.load_local(str(p), embedder, allow_dangerous_deserialization=True)
    logger.info(f"index loaded from {p}")
    return store


def get_embedder() -> EmbedderWrapper:
    return EmbedderWrapper(settings.embedding_model, settings.embedding_device)
