"""
Entry point: python -m edadvisor.ingestion.pipeline

Runs:
  load corpus → chunk → embed → build FAISS index → save
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from edadvisor.config import settings
from edadvisor.ingestion.chunkers import split
from edadvisor.ingestion.loaders import load_directory
from edadvisor.ingestion.store import build_index, get_embedder, save_index


def run(
    corpus_dir: str | None = None,
    strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> None:
    corpus_dir = corpus_dir or settings.corpus_dir
    strategy   = strategy   or settings.chunk_strategy
    chunk_size  = chunk_size  or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    corpus_path = Path(corpus_dir)
    if not any(corpus_path.iterdir()):
        logger.error(
            f"corpus directory is empty: {corpus_path}\n"
            "Add PDF / HTML / DOCX files to data/corpus/ before ingesting."
        )
        sys.exit(1)

    logger.info(f"ingestion starting  corpus={corpus_path}  strategy={strategy}")

    docs = load_directory(corpus_path)
    if not docs:
        logger.error("no documents loaded — check file formats")
        sys.exit(1)

    chunks = split(docs, strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    embedder = get_embedder()
    store = build_index(chunks, embedder)
    save_index(store, settings.vector_store_path)

    logger.info(
        f"ingestion complete  "
        f"docs={len(docs)}  chunks={len(chunks)}  "
        f"index_path={settings.vector_store_path}"
    )


if __name__ == "__main__":
    run()
