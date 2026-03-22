import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from langchain_core.documents import Document
from edadvisor.ingestion.chunkers import recursive_split, hierarchical_split, split


def _docs(texts):
    return [Document(page_content=t, metadata={"source": "test.pdf", "doc_type": "pdf"}) for t in texts]


class TestRecursiveSplit:

    def test_splits_long_doc(self):
        text = "word " * 500
        chunks = recursive_split(_docs([text]), chunk_size=200)
        assert len(chunks) > 1

    def test_short_doc_stays_one_chunk(self):
        text = "This is a short document."
        chunks = recursive_split(_docs([text]), chunk_size=500)
        assert len(chunks) == 1

    def test_chunk_ids_assigned(self):
        chunks = recursive_split(_docs(["hello " * 200]), chunk_size=100)
        assert all("chunk_id" in c.metadata for c in chunks)

    def test_metadata_preserved(self):
        chunks = recursive_split(_docs(["test " * 100]), chunk_size=50)
        assert all(c.metadata.get("source") == "test.pdf" for c in chunks)


class TestHierarchicalSplit:

    def test_creates_parent_and_child_chunks(self):
        text = "# Introduction\n\nThis is the intro section with enough words to fill it.\n\n# Requirements\n\nYou need a 2:1 degree or above in a relevant subject.\n"
        chunks = hierarchical_split(_docs([text]))
        types = {c.metadata.get("chunk_type") for c in chunks}
        assert "parent" in types
        assert "child" in types

    def test_children_link_to_parents(self):
        text = "# Section One\n\n" + "word " * 100
        chunks = hierarchical_split(_docs([text]))
        parents  = [c for c in chunks if c.metadata.get("chunk_type") == "parent"]
        children = [c for c in chunks if c.metadata.get("chunk_type") == "child"]
        parent_ids = {p.metadata["chunk_id"] for p in parents}
        for child in children:
            assert child.metadata.get("parent_id") in parent_ids

    def test_no_empty_chunks(self):
        chunks = hierarchical_split(_docs(["# Title\n\n" + "content " * 50]))
        assert all(c.page_content.strip() for c in chunks)


class TestSplitDispatch:

    def test_hierarchical_strategy(self):
        chunks = split(_docs(["# H\n\n" + "t " * 200]), strategy="hierarchical")
        assert len(chunks) > 0

    def test_recursive_strategy(self):
        chunks = split(_docs(["t " * 300]), strategy="recursive")
        assert len(chunks) > 0
