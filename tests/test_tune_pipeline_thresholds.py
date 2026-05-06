"""Tests for held-out threshold tuning script."""

import json

from src.scripts.tune_pipeline_thresholds import (
    _grid_sweep,
    _resolve_prediction,
    _stable_split,
)


def _record(claim_id, gold, baseline, baseline_conf, probs):
    return {
        "claim_id": claim_id,
        "gold_label": gold,
        "baseline_verdict": baseline,
        "baseline_confidence": baseline_conf,
        "verifier_probabilities": probs,
        "verifier_label": max(probs, key=probs.get),
    }


def test_stable_split_is_deterministic():
    records = [_record(i, "SUPPORTED", "SUPPORTED", 0.8, {"SUPPORTED": 0.7, "REFUTED": 0.2, "NOT_ENOUGH_INFO": 0.1}) for i in range(200)]
    tune_a = _stable_split(records, "tune")
    tune_b = _stable_split(records, "tune")
    holdout = _stable_split(records, "holdout")

    assert [r["claim_id"] for r in tune_a] == [r["claim_id"] for r in tune_b]
    assert len(tune_a) + len(holdout) == len(records)
    tune_ids = {r["claim_id"] for r in tune_a}
    holdout_ids = {r["claim_id"] for r in holdout}
    assert tune_ids.isdisjoint(holdout_ids)
    # Roughly balanced split.
    assert 80 <= len(tune_a) <= 120


def test_resolve_prediction_default_no_thresholds_returns_argmax():
    record = _record(
        "c1", "REFUTED", "REFUTED", 0.75,
        {"SUPPORTED": 0.10, "REFUTED": 0.60, "NOT_ENOUGH_INFO": 0.30},
    )
    label = _resolve_prediction(
        record,
        supported_threshold=0.0,
        refuted_threshold=0.0,
        min_margin=0.0,
        use_baseline_fallback=False,
        ensemble_baseline_refute_threshold=0.0,
        ensemble_verifier_refute_prob_threshold=0.0,
    )
    assert label == "REFUTED"


def test_resolve_prediction_strict_threshold_downgrades_to_nei():
    record = _record(
        "c1", "REFUTED", "REFUTED", 0.50,
        {"SUPPORTED": 0.40, "REFUTED": 0.55, "NOT_ENOUGH_INFO": 0.05},
    )
    label = _resolve_prediction(
        record,
        supported_threshold=0.99,
        refuted_threshold=0.99,
        min_margin=0.0,
        use_baseline_fallback=False,
        ensemble_baseline_refute_threshold=0.0,
        ensemble_verifier_refute_prob_threshold=0.0,
    )
    assert label == "NOT_ENOUGH_INFO"


def test_resolve_prediction_baseline_fallback_recovers_refuted():
    record = _record(
        "c1", "REFUTED", "REFUTED", 0.90,
        # Verifier says NEI but refute prob is non-trivial; baseline says
        # REFUTED with high confidence — fallback should recover it.
        {"SUPPORTED": 0.10, "REFUTED": 0.30, "NOT_ENOUGH_INFO": 0.60},
    )
    label = _resolve_prediction(
        record,
        supported_threshold=0.0,
        refuted_threshold=0.0,
        min_margin=0.0,
        use_baseline_fallback=True,
        ensemble_baseline_refute_threshold=0.80,
        ensemble_verifier_refute_prob_threshold=0.25,
    )
    assert label == "REFUTED"


def test_grid_sweep_picks_thresholds_that_match_perfect_records():
    """When records are 100% separable by argmax, sweep should achieve macro F1 = 1."""
    records = []
    for i in range(30):
        records.append(_record(
            i, "SUPPORTED", "SUPPORTED", 0.9,
            {"SUPPORTED": 0.8, "REFUTED": 0.1, "NOT_ENOUGH_INFO": 0.1},
        ))
    for i in range(30):
        records.append(_record(
            100 + i, "REFUTED", "REFUTED", 0.9,
            {"SUPPORTED": 0.05, "REFUTED": 0.85, "NOT_ENOUGH_INFO": 0.10},
        ))
    for i in range(30):
        records.append(_record(
            200 + i, "NOT_ENOUGH_INFO", "NOT_ENOUGH_INFO", 0.5,
            {"SUPPORTED": 0.1, "REFUTED": 0.1, "NOT_ENOUGH_INFO": 0.8},
        ))

    best = _grid_sweep(records)

    assert best["macro_f1"] >= 0.99
    assert best["thresholds"] is not None
