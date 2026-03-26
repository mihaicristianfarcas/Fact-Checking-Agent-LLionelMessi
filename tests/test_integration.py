"""Integration tests for the full data ingestion pipeline.

These tests require downloading datasets and may take longer to run.
Run with: pytest tests/test_integration.py -v
"""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_index_dir():
    """Create temporary directory for ChromaDB index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestPreprocessingIntegration:
    """Integration tests for preprocessing pipeline."""

    def test_full_preprocessing_pipeline(self):
        """Test text cleaning and chunking together."""
        from src.data_ingestion.preprocessing import (
            TextCleaner,
            TextChunker,
            SentenceSplitter,
        )

        raw_text = """
        <p>The Earth is approximately 4.5 billion years old. This was determined
        through radiometric dating of meteorites. Scientists have studied many
        samples to confirm this age estimate.</p>
        """

        # Clean
        cleaner = TextCleaner()
        clean = cleaner.clean(raw_text)
        assert "<p>" not in clean
        assert "Earth" in clean

        # Split into sentences
        splitter = SentenceSplitter(min_length=20)
        sentences = splitter.split(clean)
        assert len(sentences) >= 2

        # Chunk for indexing
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(clean)
        assert len(chunks) >= 1


class TestDataModelsIntegration:
    """Test data model serialization round-trips."""

    def test_claim_full_roundtrip(self):
        """Test complete Claim serialization."""
        from src.data_ingestion.datasets.base import (
            Claim,
            ClaimEvidenceTriple,
            EvidencePassage,
            Verdict,
        )
        import json

        # Create complex claim
        evidence = [
            EvidencePassage(
                id="ev_1",
                text="Evidence sentence one.",
                source="Wikipedia:Earth",
                dataset="fever",
                metadata={"sentence_id": 0},
            ),
            EvidencePassage(
                id="ev_2",
                text="Evidence sentence two.",
                source="Wikipedia:Earth",
                dataset="fever",
                metadata={"sentence_id": 1},
            ),
        ]

        claim = Claim(
            id="test_claim_1",
            text="The Earth is round.",
            verdict=Verdict.SUPPORTED,
            evidence=evidence,
            dataset="fever",
            metadata={"source": "test"},
        )

        # Serialize to JSON and back
        json_str = json.dumps(claim.to_dict())
        restored_data = json.loads(json_str)
        restored = Claim.from_dict(restored_data)

        assert restored.id == claim.id
        assert restored.verdict == claim.verdict
        assert len(restored.evidence) == 2
        assert restored.evidence[0].text == "Evidence sentence one."

    def test_triple_json_export(self):
        """Test triple JSON export format."""
        from src.data_ingestion.datasets.base import (
            ClaimEvidenceTriple,
            EvidencePassage,
            Verdict,
        )
        import json

        triple = ClaimEvidenceTriple(
            claim_id="c1",
            claim_text="Test claim",
            evidence_passages=[
                EvidencePassage(
                    id="e1", text="Evidence", source="wiki", dataset="fever"
                )
            ],
            verdict=Verdict.SUPPORTED,
            confidence=0.95,
            metadata={"dataset": "fever", "split": "train"},
        )

        # Should produce valid JSON for fine-tuning
        data = triple.to_dict()
        json_str = json.dumps(data, indent=2)
        assert '"verdict": "SUPPORTED"' in json_str
        assert '"confidence": 0.95' in json_str


@pytest.mark.skip(reason="Requires model download (~90MB)")
class TestEmbeddingIntegration:
    """Integration tests for embedding generation."""

    def test_embedder_basic(self):
        """Test basic embedding generation."""
        from src.data_ingestion.indexing import Embedder

        embedder = Embedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=8,
        )

        texts = ["Hello world", "Test sentence"]
        embeddings = embedder.embed(texts, show_progress=False)

        assert embeddings.shape == (2, 384)

    def test_embedder_single(self):
        """Test single text embedding."""
        from src.data_ingestion.indexing import Embedder

        embedder = Embedder()
        embedding = embedder.embed_single("Test text")
        assert embedding.shape == (384,)


@pytest.mark.skip(reason="Requires model download and ChromaDB setup")
class TestIndexIntegration:
    """Integration tests for ChromaDB indexing."""

    def test_index_and_search(self, temp_index_dir):
        """Test indexing and retrieval."""
        from src.data_ingestion.datasets.base import EvidencePassage
        from src.data_ingestion.indexing import ChromaIndex

        index = ChromaIndex(
            persist_dir=temp_index_dir / "chroma",
            collection_name="test_collection",
        )

        # Add passages
        passages = [
            EvidencePassage(
                id="p1",
                text="The Earth orbits the Sun.",
                source="astronomy",
                dataset="test",
            ),
            EvidencePassage(
                id="p2",
                text="Water is composed of hydrogen and oxygen.",
                source="chemistry",
                dataset="test",
            ),
        ]
        index.add_passages(passages)

        # Search
        results = index.search("planets in solar system", top_k=2)
        assert len(results) > 0
        # Earth/Sun result should be more relevant
        assert results[0]["id"] == "p1"

    def test_index_stats(self, temp_index_dir):
        """Test index statistics."""
        from src.data_ingestion.indexing import ChromaIndex

        index = ChromaIndex(persist_dir=temp_index_dir / "chroma")
        stats = index.get_stats()
        assert "total_documents" in stats
        assert "embedding_dimension" in stats
