from __future__ import annotations

import re

from langchain_core.documents import Document


def _format_context(docs: list[Document]) -> str:
    """
    Format retrieved documents into the context string injected into the prompt.
    Each chunk is labelled [Source N] so the LLM can cite it inline.
    """
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", f"document_{i}")
        heading = doc.metadata.get("section_heading", "")
        header = f"[Source {i}] {source}"
        if heading:
            header += f" — {heading}"
        parts.append(f"{header}\n{doc.page_content[:600]}")
    return "\n\n---\n\n".join(parts)


def extract_citations(answer: str, docs: list[Document]) -> list[dict]:
    """
    Parse [Source N] references from the LLM answer and map each back
    to its actual document metadata.

    Returns a list of dicts: {source, excerpt, page, doc_type}
    """
    mentioned = set(map(int, re.findall(r"\[Source (\d+)\]", answer)))
    results = []
    for n in sorted(mentioned):
        idx = n - 1
        if 0 <= idx < len(docs):
            doc = docs[idx]
            results.append({
                "source_n": n,
                "source": doc.metadata.get("source", "unknown"),
                "excerpt": doc.page_content[:150].replace("\n", " "),
                "page": doc.metadata.get("page"),
                "doc_type": doc.metadata.get("doc_type", ""),
                "section": doc.metadata.get("section_heading", ""),
            })
    return results
