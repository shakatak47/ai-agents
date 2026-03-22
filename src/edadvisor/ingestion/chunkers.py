from __future__ import annotations

import re
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


def _slug(text: str, max_len: int = 60) -> str:
    """Turn a heading into a short readable id."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower().strip())[:max_len].strip("-")


# ─── Strategy 1: simple recursive split ────────────────────────────────────

def recursive_split(
    docs: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " "],
    )
    chunks = splitter.split_documents(docs)
    # tag every chunk with a stable id
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = f"rc-{i:05d}"
        c.metadata["chunk_index"] = i
    logger.info(f"recursive split  {len(docs)} docs → {len(chunks)} chunks")
    return chunks


# ─── Strategy 2: hierarchical — preserve heading structure ──────────────────

_HEADING_RE = re.compile(r"^(#{1,3}|[A-Z][A-Z ]{3,})\s+(.+)$", re.MULTILINE)


def _detect_sections(text: str) -> list[tuple[str, str]]:
    """
    Split text into (heading, body) pairs.
    Works on markdown headings and ALL-CAPS section titles.
    Falls back to a single section if no headings found.
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("main", text)]

    sections = []
    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((heading, body))
    return sections


def hierarchical_split(
    docs: list[Document],
    parent_size: int = 1500,
    child_size: int = 400,
    overlap: int = 80,
) -> list[Document]:
    """
    Two-level chunking: large parent chunks (section-level) +
    small child chunks (paragraph-level) linked by parent_id.

    At retrieval time we match on child chunks but return
    the parent context — better coherence for multi-part answers.
    """
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=overlap,
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=100,
    )

    all_chunks: list[Document] = []

    for doc in docs:
        sections = _detect_sections(doc.page_content)
        for heading, body in sections:
            parent_id = str(uuid.uuid4())[:8]

            # parent chunk
            parent_doc = Document(
                page_content=body[:parent_size],
                metadata={
                    **doc.metadata,
                    "chunk_id": parent_id,
                    "chunk_type": "parent",
                    "section_heading": heading,
                },
            )
            all_chunks.append(parent_doc)

            # child chunks within this section
            children = child_splitter.create_documents(
                texts=[body],
                metadatas=[{
                    **doc.metadata,
                    "parent_id": parent_id,
                    "chunk_type": "child",
                    "section_heading": heading,
                }],
            )
            for j, child in enumerate(children):
                child.metadata["chunk_id"] = f"{parent_id}-{j:03d}"
                child.metadata["chunk_index"] = j
            all_chunks.extend(children)

    parents = sum(1 for c in all_chunks if c.metadata.get("chunk_type") == "parent")
    children = len(all_chunks) - parents
    logger.info(f"hierarchical split  {len(docs)} docs → {parents} parents + {children} children")
    return all_chunks


def split(
    docs: list[Document],
    strategy: str = "hierarchical",
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Document]:
    if strategy == "hierarchical":
        return hierarchical_split(docs, parent_size=chunk_size * 2, child_size=chunk_size // 2)
    return recursive_split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
