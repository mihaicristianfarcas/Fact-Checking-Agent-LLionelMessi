"""Tests for offline evaluation metrics."""

import json

from src.evaluation.metrics import (
    EvaluationRecord,
    audit_citations,
    evaluate_records,
    expected_calibration_error,
    load_prediction_records,
    normalize_label,
)


def test_normalize_label_accepts_fever_labels():
    assert normalize_label("SUPPORTS") == "SUPPORTED"
    assert normalize_label("REFUTES") == "REFUTED"
    assert normalize_label("NOT ENOUGH INFO") == "NOT_ENOUGH_INFO"


def test_expected_calibration_error_includes_confidence_one():
    ece = expected_calibration_error([1.0, 0.0], [True, False], n_bins=2)
    assert ece == 0.0


def test_audit_citations_flags_hallucinated_ids():
    audit = audit_citations(
        cited_passage_ids=["p1", "made_up"],
        retrieved_passage_ids=["p1", "p2"],
        verdict="SUPPORTED",
    )

    assert audit.hallucinated_citations == ["made_up"]
    assert audit.has_violation is True


def test_audit_citations_requires_citation_for_decisive_verdict():
    audit = audit_citations(
        cited_passage_ids=[],
        retrieved_passage_ids=["p1"],
        verdict="REFUTED",
    )

    assert audit.missing_required_citation is True


def test_evaluate_records_reports_verdict_and_citation_metrics():
    records = [
        EvaluationRecord(
            claim_id="1",
            gold_verdict="SUPPORTED",
            predicted_verdict="SUPPORTED",
            confidence=0.9,
            cited_passage_ids=["p1"],
            retrieved_passage_ids=["p1", "p2"],
        ),
        EvaluationRecord(
            claim_id="2",
            gold_verdict="REFUTED",
            predicted_verdict="SUPPORTED",
            confidence=0.7,
            cited_passage_ids=["fake"],
            retrieved_passage_ids=["p3"],
        ),
        EvaluationRecord(
            claim_id="3",
            gold_verdict="NOT_ENOUGH_INFO",
            predicted_verdict="NOT_ENOUGH_INFO",
            confidence=0.8,
            cited_passage_ids=[],
            retrieved_passage_ids=[],
        ),
    ]

    metrics = evaluate_records(records)

    assert metrics["n_claims"] == 3
    assert metrics["accuracy"] == 2 / 3
    assert metrics["hallucinated_citation_count"] == 1
    assert metrics["citation_violation_count"] == 1
    assert metrics["per_class"]["SUPPORTED"]["support"] == 1


def test_load_prediction_records_from_jsonl(tmp_path):
    path = tmp_path / "predictions.jsonl"
    row = {
        "claim_id": "a",
        "gold_verdict": "SUPPORTED",
        "predicted_verdict": "SUPPORTED",
        "confidence": 0.75,
        "cited_passage_ids": ["p1"],
        "retrieved_passage_ids": ["p1"],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    records = load_prediction_records(path)

    assert len(records) == 1
    assert records[0].claim_id == "a"
