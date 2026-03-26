"""Tests for triple generation."""

import pytest

from src.data_ingestion.datasets.base import (
    ClaimEvidenceTriple,
    EvidencePassage,
    Verdict,
)
from src.data_ingestion.triples.triple_generator import TripleGenerator


class TestTripleGenerator:
    """Test TripleGenerator class."""

    def test_init(self):
        generator = TripleGenerator()
        assert len(generator.get_triples()) == 0

    def test_get_statistics_empty(self):
        generator = TripleGenerator()
        stats = generator.get_statistics()
        assert stats["total"] == 0

    def test_iter_triples(self):
        generator = TripleGenerator()
        # Manually add a triple for testing
        triple = ClaimEvidenceTriple(
            claim_id="test_1",
            claim_text="Test claim",
            evidence_passages=[],
            verdict=Verdict.SUPPORTED,
        )
        generator._triples.append(triple)

        triples = list(generator.iter_triples())
        assert len(triples) == 1
        assert triples[0].claim_id == "test_1"

    def test_get_statistics_with_data(self):
        generator = TripleGenerator()

        # Add test triples
        for verdict in [Verdict.SUPPORTED, Verdict.SUPPORTED, Verdict.REFUTED]:
            generator._triples.append(
                ClaimEvidenceTriple(
                    claim_id=f"test_{verdict.value}",
                    claim_text="Test",
                    evidence_passages=[],
                    verdict=verdict,
                    metadata={"dataset": "test"},
                )
            )

        stats = generator.get_statistics()
        assert stats["total"] == 3
        assert stats["verdict_distribution"]["SUPPORTED"] == 2
        assert stats["verdict_distribution"]["REFUTED"] == 1


# Integration tests
@pytest.mark.skip(reason="Requires dataset download")
class TestTripleGeneratorIntegration:
    """Integration tests requiring dataset downloads."""

    def test_load_fever(self):
        generator = TripleGenerator()
        count = generator.load_fever(split="labelled_dev", max_samples=100)
        assert count > 0

    def test_load_politifact(self):
        generator = TripleGenerator()
        count = generator.load_politifact(max_samples=100)
        assert count > 0
