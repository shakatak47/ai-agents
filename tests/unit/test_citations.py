import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from langchain_core.documents import Document
from edadvisor.generation.citations import _format_context, extract_citations


def _doc(content, source="test.pdf", page=None, section=""):
    meta = {"source": source, "doc_type": "pdf", "section_heading": section}
    if page:
        meta["page"] = page
    return Document(page_content=content, metadata=meta)


class TestFormatContext:

    def test_labels_sources_numerically(self):
        docs = [_doc("content A"), _doc("content B")]
        ctx = _format_context(docs)
        assert "[Source 1]" in ctx
        assert "[Source 2]" in ctx

    def test_includes_source_name(self):
        docs = [_doc("text", source="visa_guide.pdf")]
        ctx = _format_context(docs)
        assert "visa_guide.pdf" in ctx

    def test_includes_section_heading(self):
        docs = [_doc("text", section="Visa Requirements")]
        ctx = _format_context(docs)
        assert "Visa Requirements" in ctx

    def test_empty_docs_returns_empty_string(self):
        assert _format_context([]) == ""

    def test_content_truncated_to_600_chars(self):
        long_text = "x " * 500
        docs = [_doc(long_text)]
        ctx = _format_context(docs)
        # each source block should not contain the full long text
        assert len(ctx) < len(long_text) * 2


class TestExtractCitations:

    def test_extracts_referenced_sources(self):
        docs = [_doc("IELTS requirement is 6.5"), _doc("TOEFL accepted")]
        answer = "You need IELTS 6.5 [Source 1]. TOEFL is also accepted [Source 2]."
        cites = extract_citations(answer, docs)
        source_ns = [c["source_n"] for c in cites]
        assert 1 in source_ns
        assert 2 in source_ns

    def test_unreferenced_sources_not_included(self):
        docs = [_doc("A"), _doc("B"), _doc("C")]
        answer = "Only [Source 1] was relevant."
        cites = extract_citations(answer, docs)
        assert len(cites) == 1
        assert cites[0]["source_n"] == 1

    def test_no_citations_returns_empty(self):
        docs = [_doc("content")]
        cites = extract_citations("No sources cited here.", docs)
        assert cites == []

    def test_out_of_range_source_ignored(self):
        docs = [_doc("only one doc")]
        answer = "See [Source 99] for details."
        cites = extract_citations(answer, docs)
        assert cites == []

    def test_excerpt_is_truncated(self):
        long_content = "word " * 100
        docs = [_doc(long_content)]
        answer = "Answer [Source 1]"
        cites = extract_citations(answer, docs)
        assert len(cites[0]["excerpt"]) <= 155
