"""Tests for dataset loaders."""

import pytest

from src.data_ingestion.datasets.base import (
    Claim,
    ClaimEvidenceTriple,
    EvidencePassage,
    Verdict,
)


class TestVerdict:
    """Test Verdict enum."""

    def test_verdict_values(self):
        assert Verdict.SUPPORTED.value == "SUPPORTED"
        assert Verdict.REFUTED.value == "REFUTED"
        assert Verdict.NOT_ENOUGH_INFO.value == "NOT_ENOUGH_INFO"

    def test_verdict_from_string(self):
        assert Verdict("SUPPORTED") == Verdict.SUPPORTED
        assert Verdict("REFUTED") == Verdict.REFUTED


class TestEvidencePassage:
    """Test EvidencePassage dataclass."""

    def test_creation(self):
        passage = EvidencePassage(
            id="test_1",
            text="Test evidence text",
            source="Wikipedia",
            dataset="fever",
            metadata={"sentence_id": 0},
        )
        assert passage.id == "test_1"
        assert passage.text == "Test evidence text"
        assert passage.source == "Wikipedia"

    def test_to_dict(self):
        passage = EvidencePassage(
            id="test_1",
            text="Test text",
            source="wiki",
            dataset="fever",
        )
        d = passage.to_dict()
        assert d["id"] == "test_1"
        assert d["dataset"] == "fever"

    def test_from_dict(self):
        data = {
            "id": "test_1",
            "text": "Test text",
            "source": "wiki",
            "dataset": "fever",
            "metadata": {"key": "value"},
        }
        passage = EvidencePassage.from_dict(data)
        assert passage.id == "test_1"
        assert passage.metadata["key"] == "value"


class TestClaim:
    """Test Claim dataclass."""

    def test_creation(self):
        claim = Claim(
            id="claim_1",
            text="The sky is blue.",
            verdict=Verdict.SUPPORTED,
            dataset="fever",
        )
        assert claim.id == "claim_1"
        assert claim.verdict == Verdict.SUPPORTED

    def test_claim_with_evidence(self):
        evidence = EvidencePassage(
            id="ev_1", text="Evidence text", source="wiki", dataset="fever"
        )
        claim = Claim(
            id="claim_1",
            text="Claim text",
            verdict=Verdict.REFUTED,
            evidence=[evidence],
            dataset="fever",
        )
        assert len(claim.evidence) == 1
        assert claim.evidence[0].text == "Evidence text"

    def test_to_dict_and_back(self):
        claim = Claim(
            id="claim_1",
            text="Test claim",
            verdict=Verdict.NOT_ENOUGH_INFO,
            dataset="test",
        )
        d = claim.to_dict()
        restored = Claim.from_dict(d)
        assert restored.id == claim.id
        assert restored.verdict == claim.verdict


class TestClaimEvidenceTriple:
    """Test ClaimEvidenceTriple dataclass."""

    def test_creation(self):
        triple = ClaimEvidenceTriple(
            claim_id="c1",
            claim_text="Test claim",
            evidence_passages=[],
            verdict=Verdict.SUPPORTED,
            confidence=0.95,
        )
        assert triple.claim_id == "c1"
        assert triple.confidence == 0.95

    def test_serialization(self):
        evidence = EvidencePassage(
            id="e1", text="Evidence", source="wiki", dataset="fever"
        )
        triple = ClaimEvidenceTriple(
            claim_id="c1",
            claim_text="Claim",
            evidence_passages=[evidence],
            verdict=Verdict.REFUTED,
        )

        d = triple.to_dict()
        assert d["verdict"] == "REFUTED"
        assert len(d["evidence_passages"]) == 1

        restored = ClaimEvidenceTriple.from_dict(d)
        assert restored.verdict == Verdict.REFUTED
