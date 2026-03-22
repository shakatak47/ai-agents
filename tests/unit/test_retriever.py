import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from edadvisor.retrieval.retriever import _confidence_from_scores, retrieve, RetrievalResult


class TestConfidenceScore:

    def test_empty_scores_returns_zero(self):
        assert _confidence_from_scores([], threshold=0.7) == 0.0

    def test_all_above_threshold_high_confidence(self):
        scores = [0.9, 0.85, 0.88, 0.91]
        conf = _confidence_from_scores(scores, threshold=0.7)
        assert conf > 0.7

    def test_all_below_threshold_low_confidence(self):
        scores = [0.4, 0.3, 0.5]
        conf = _confidence_from_scores(scores, threshold=0.7)
        assert conf < 0.5

    def test_returns_float(self):
        conf = _confidence_from_scores([0.8, 0.75], threshold=0.7)
        assert isinstance(conf, float)

    def test_partial_above_threshold(self):
        scores = [0.9, 0.85, 0.4, 0.3]  # 2 of 4 above threshold
        conf = _confidence_from_scores(scores, threshold=0.7)
        # coverage penalty: only 50% above threshold
        assert conf < _confidence_from_scores([0.9, 0.85, 0.88, 0.87], threshold=0.7)


class TestRetrieve:

    def _mock_store(self, docs_and_scores=None):
        from langchain_core.documents import Document
        store = MagicMock()
        if docs_and_scores is None:
            docs_and_scores = [
                (Document(page_content="IELTS 6.5 required", metadata={"source": "guide.pdf"}), 0.85),
                (Document(page_content="TOEFL 90 accepted", metadata={"source": "guide.pdf"}), 0.80),
            ]
        store.similarity_search_with_score.return_value = docs_and_scores
        store.max_marginal_relevance_search.return_value = [d for d, _ in docs_and_scores]
        return store

    def test_returns_retrieval_result(self):
        store = self._mock_store()
        result = retrieve("What is the IELTS requirement?", store, threshold=0.7)
        assert isinstance(result, RetrievalResult)

    def test_high_score_does_not_escalate(self):
        store = self._mock_store()
        result = retrieve("IELTS score needed?", store, threshold=0.7)
        assert result.should_escalate is False
        assert result.confidence > 0

    def test_low_scores_trigger_escalation(self):
        from langchain_core.documents import Document
        low_results = [
            (Document(page_content="something irrelevant", metadata={}), 0.3),
        ]
        store = self._mock_store(low_results)
        result = retrieve("quantum physics question", store, threshold=0.7)
        assert result.should_escalate is True

    def test_empty_store_returns_escalation(self):
        store = self._mock_store([])
        result = retrieve("anything", store, threshold=0.7)
        assert result.should_escalate is True
        assert result.docs == []

    def test_faiss_error_returns_escalation(self):
        store = MagicMock()
        store.similarity_search_with_score.side_effect = RuntimeError("FAISS error")
        result = retrieve("question", store)
        assert result.should_escalate is True
        assert result.error is None or result.docs == []
