"""Pure-Python evaluation metrics for fact-checking outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.model_training.prompting import (
    NOT_ENOUGH_INFO,
    REFUTED,
    SUPPORTED,
    normalize_verdict,
)

LABELS = (SUPPORTED, REFUTED, NOT_ENOUGH_INFO)


@dataclass(frozen=True)
class CitationAudit:
    """Citation quality checks for one predicted verdict."""

    hallucinated_citations: list[str]
    missing_required_citation: bool

    @property
    def has_violation(self) -> bool:
        return bool(self.hallucinated_citations) or self.missing_required_citation


@dataclass(frozen=True)
class EvaluationRecord:
    """One prediction row consumed by the evaluation harness."""

    claim_id: str
    gold_verdict: str
    predicted_verdict: str
    confidence: float = 0.0
    cited_passage_ids: list[str] = field(default_factory=list)
    retrieved_passage_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "EvaluationRecord":
        gold = (
            row.get("gold_verdict")
            or row.get("gold_label")
            or row.get("label")
            or row.get("expected_verdict")
        )
        pred = (
            row.get("predicted_verdict")
            or row.get("predicted_label")
            or row.get("prediction")
            or row.get("model_verdict")
        )
        if gold is None or pred is None:
            raise ValueError(
                "Each prediction row must include a gold label and a predicted label."
            )

        gold_norm = normalize_label(gold)
        pred_norm = normalize_label(pred)

        return cls(
            claim_id=str(row.get("claim_id") or row.get("id") or ""),
            gold_verdict=gold_norm,
            predicted_verdict=pred_norm,
            confidence=float(row.get("confidence", 0.0) or 0.0),
            cited_passage_ids=_listify(
                row.get("cited_passage_ids")
                or row.get("citations")
                or row.get("cited_ids")
            ),
            retrieved_passage_ids=_listify(
                row.get("retrieved_passage_ids")
                or row.get("retrieved_ids")
                or row.get("all_retrieved_ids")
            ),
        )


def normalize_label(label: str) -> str:
    """Normalize labels and fail loudly on unknown classes."""
    normalized = normalize_verdict(label)
    if normalized not in LABELS:
        raise ValueError(f"Unknown verdict label: {label!r}")
    return normalized


def audit_citations(
    cited_passage_ids: Sequence[str],
    retrieved_passage_ids: Sequence[str],
    verdict: str,
) -> CitationAudit:
    """Check that every citation came from retrieval and verdicts cite evidence."""
    retrieved = {str(pid) for pid in retrieved_passage_ids}
    hallucinated = [
        str(pid) for pid in cited_passage_ids if str(pid) not in retrieved
    ]
    missing_required = (
        normalize_label(verdict) != NOT_ENOUGH_INFO and not cited_passage_ids
    )
    return CitationAudit(
        hallucinated_citations=hallucinated,
        missing_required_citation=missing_required,
    )


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """Compute weighted ECE using fixed-width confidence bins."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length.")
    if not confidences:
        return 0.0
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    total = len(confidences)
    ece = 0.0
    for bin_idx in range(n_bins):
        lo = bin_idx / n_bins
        hi = (bin_idx + 1) / n_bins
        members = [
            (conf, is_correct)
            for conf, is_correct in zip(confidences, correct)
            if (lo <= conf < hi) or (bin_idx == n_bins - 1 and conf == 1.0)
        ]
        if not members:
            continue
        avg_conf = sum(conf for conf, _ in members) / len(members)
        avg_acc = sum(1.0 for _, is_correct in members if is_correct) / len(members)
        ece += (len(members) / total) * abs(avg_conf - avg_acc)
    return ece


def evaluate_records(
    records: Iterable[EvaluationRecord | Mapping[str, Any]],
    *,
    labels: Sequence[str] = LABELS,
    hallucination_target: float = 0.05,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate verdict quality, calibration, and citation faithfulness."""
    normalized_records = [
        row if isinstance(row, EvaluationRecord) else EvaluationRecord.from_mapping(row)
        for row in records
    ]
    n = len(normalized_records)
    if n == 0:
        return _empty_result(labels, hallucination_target)

    y_true = [record.gold_verdict for record in normalized_records]
    y_pred = [record.predicted_verdict for record in normalized_records]
    confidences = [record.confidence for record in normalized_records]
    correct = [gold == pred for gold, pred in zip(y_true, y_pred)]

    confusion = _confusion_matrix(y_true, y_pred, labels)
    per_class = {
        label: _class_metrics(confusion, labels, label)
        for label in labels
    }

    audits = [
        audit_citations(
            record.cited_passage_ids,
            record.retrieved_passage_ids,
            record.predicted_verdict,
        )
        for record in normalized_records
    ]
    hallucination_count = sum(1 for audit in audits if audit.hallucinated_citations)
    missing_citation_count = sum(
        1 for audit in audits if audit.missing_required_citation
    )
    citation_violation_count = sum(1 for audit in audits if audit.has_violation)

    accuracy = sum(1 for flag in correct if flag) / n
    macro_f1 = sum(per_class[label]["f1"] for label in labels) / len(labels)
    ece = expected_calibration_error(confidences, correct, n_bins=calibration_bins)

    return {
        "n_claims": n,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "ece": ece,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "hallucinated_citation_count": hallucination_count,
        "hallucination_rate": hallucination_count / n,
        "missing_citation_count": missing_citation_count,
        "missing_citation_rate": missing_citation_count / n,
        "citation_violation_count": citation_violation_count,
        "citation_violation_rate": citation_violation_count / n,
        "hallucination_target": hallucination_target,
        "hallucination_target_met": (hallucination_count / n) < hallucination_target,
    }


def load_prediction_records(path: str | Path) -> list[EvaluationRecord]:
    """Load prediction rows from JSONL or a JSON list."""
    input_path = Path(path)
    if input_path.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        loaded = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError("JSON prediction files must contain a list of rows.")
        rows = loaded
    return [EvaluationRecord.from_mapping(row) for row in rows]


def _empty_result(
    labels: Sequence[str],
    hallucination_target: float,
) -> dict[str, Any]:
    return {
        "n_claims": 0,
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "ece": 0.0,
        "per_class": {
            label: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
            for label in labels
        },
        "confusion_matrix": {
            label: {predicted: 0 for predicted in labels}
            for label in labels
        },
        "hallucinated_citation_count": 0,
        "hallucination_rate": 0.0,
        "missing_citation_count": 0,
        "missing_citation_rate": 0.0,
        "citation_violation_count": 0,
        "citation_violation_rate": 0.0,
        "hallucination_target": hallucination_target,
        "hallucination_target_met": True,
    }


def _class_metrics(
    confusion: Mapping[str, Mapping[str, int]],
    labels: Sequence[str],
    label: str,
) -> dict[str, float | int]:
    tp = confusion[label][label]
    fp = sum(confusion[actual][label] for actual in labels if actual != label)
    fn = sum(confusion[label][predicted] for predicted in labels if predicted != label)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    support = sum(confusion[label][predicted] for predicted in labels)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def _confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> dict[str, dict[str, int]]:
    matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual][predicted] += 1
    return matrix


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]
