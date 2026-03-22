"""
Seed the corpus with publicly available education guidance pages.
Run this once to populate data/corpus/ before running `make ingest`.

Usage:
    python scripts/fetch_sample_corpus.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import httpx
from loguru import logger

SOURCES = [
    {
        "url": "https://www.gov.uk/student-visa",
        "filename": "uk_student_visa.html",
        "description": "UK Student Visa official guidance",
    },
    {
        "url": "https://www.gov.uk/student-visa/documents-you-must-provide",
        "filename": "uk_student_visa_documents.html",
        "description": "UK Student Visa document requirements",
    },
    {
        "url": "https://www.chevening.org/scholarship/",
        "filename": "chevening_scholarship.html",
        "description": "Chevening Scholarship information",
    },
]

CORPUS_DIR = Path("data/corpus")


def fetch_all() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0

    for source in SOURCES:
        out_path = CORPUS_DIR / source["filename"]
        if out_path.exists():
            logger.info(f"already exists: {out_path.name} — skipping")
            continue

        logger.info(f"fetching {source['description']}…")
        try:
            resp = httpx.get(source["url"], timeout=20, follow_redirects=True)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            logger.info(f"saved {out_path.name}  ({len(resp.content):,} bytes)")
            saved += 1
        except Exception as e:
            logger.error(f"failed {source['url']}: {e}")

    logger.info(f"done  saved={saved}/{len(SOURCES)}")
    if saved > 0:
        logger.info("now run: make ingest")


if __name__ == "__main__":
    fetch_all()
