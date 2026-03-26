"""Tests for evidence retriever."""

import pytest


class TestRetrievalResult:
    """Test RetrievalResult dataclass."""

    def test_creation(self):
        from src.data_ingestion.datasets.base import EvidencePassage
        from src.data_ingestion.retriever.evidence_retriever import RetrievalResult

        passage = EvidencePassage(
            id="test_1",
            text="Test text",
            source="wiki",
            dataset="fever",
        )
        result = RetrievalResult(passage=passage, score=0.95, rank=1)

        assert result.score == 0.95
        assert result.rank == 1
        assert result.passage.id == "test_1"


class TestEvidenceRetrieverInit:
    """Test EvidenceRetriever initialization (without loading models)."""

    def test_init_defaults(self):
        from src.data_ingestion.retriever.evidence_retriever import EvidenceRetriever

        # Just test that initialization doesn't fail
        # (lazy loading means no models loaded yet)
        retriever = EvidenceRetriever(
            index_path="/tmp/test_chroma",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        assert retriever.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert not retriever._initialized


# Integration tests would require loading models and index
# Marked as skip for CI/quick testing
@pytest.mark.skip(reason="Requires model download and index")
class TestEvidenceRetrieverIntegration:
    """Integration tests for EvidenceRetriever."""

    def test_retrieve_basic(self):
        from src.data_ingestion.retriever.evidence_retriever import EvidenceRetriever

        retriever = EvidenceRetriever()
        results = retriever.retrieve("test query", top_k=5)
        assert isinstance(results, list)

    def test_retrieve_batch(self):
        from src.data_ingestion.retriever.evidence_retriever import EvidenceRetriever

        retriever = EvidenceRetriever()
        results = retriever.retrieve_batch(["query 1", "query 2"], top_k=5)
        assert len(results) == 2
