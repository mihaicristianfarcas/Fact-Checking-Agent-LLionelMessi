"""Tests for verifier temperature scaling."""

import json
import math

import pytest

from src.model_training.calibration import (
    apply_temperature,
    fit_temperature,
    load_temperature_sidecar,
    write_temperature_sidecar,
)


def test_apply_temperature_one_recovers_softmax():
    """T=1 must give the standard softmax."""
    logits = [2.0, 1.0, 0.0]
    probs = apply_temperature(logits, temperature=1.0)
    expected = [math.exp(x) for x in logits]
    s = sum(expected)
    expected = [e / s for e in expected]
    for p, e in zip(probs, expected):
        assert abs(p - e) < 1e-9


def test_apply_temperature_high_t_flattens_distribution():
    """T -> infinity drives probabilities toward uniform."""
    logits = [10.0, 0.0, 0.0]
    sharp = apply_temperature(logits, temperature=1.0)
    flat = apply_temperature(logits, temperature=100.0)
    assert sharp[0] > 0.99
    # All three classes should be near 1/3 with high T.
    for p in flat:
        assert abs(p - 1 / 3) < 0.05


def test_apply_temperature_rejects_nonpositive():
    with pytest.raises(ValueError):
        apply_temperature([1.0, 0.0], temperature=0.0)


def test_fit_temperature_corrects_overconfidence():
    """An overconfident classifier should be assigned T > 1."""
    # 3-class problem. Use overconfident logits (large margins) but the
    # *correct* class is consistently selected. The well-calibrated T will be
    # > 1 because the logit gap is too wide for the empirical accuracy.
    # Build 30 examples: 25 correct (large margin), 5 wrong (large margin).
    correct_logits = [[5.0, 0.0, 0.0]] * 25
    correct_labels = [0] * 25
    wrong_logits = [[5.0, 0.0, 0.0]] * 5
    wrong_labels = [1] * 5

    logits = correct_logits + wrong_logits
    labels = correct_labels + wrong_labels

    T = fit_temperature(logits, labels)
    assert T > 1.0, f"Expected T > 1 for overconfident logits, got {T}"


def test_fit_temperature_argmax_invariant():
    """Calibration must not change argmax predictions."""
    logits = [[3.0, 1.0, 0.0], [0.0, 2.5, 0.5], [0.5, 0.4, 3.0]]
    labels = [0, 1, 2]
    T = fit_temperature(logits, labels)

    for row in logits:
        cal = apply_temperature(row, temperature=T)
        assert cal.index(max(cal)) == row.index(max(row))


def test_fit_temperature_rejects_empty():
    with pytest.raises(ValueError):
        fit_temperature([], [])


def test_fit_temperature_rejects_length_mismatch():
    with pytest.raises(ValueError):
        fit_temperature([[1.0, 2.0]], [0, 1])


def test_temperature_sidecar_roundtrip(tmp_path):
    write_temperature_sidecar(tmp_path, 1.42, metadata={"fit_on": "dev_2k"})
    loaded = load_temperature_sidecar(tmp_path)
    assert loaded == 1.42

    raw = json.loads((tmp_path / "temperature.json").read_text())
    assert raw["temperature"] == 1.42
    assert raw["metadata"]["fit_on"] == "dev_2k"


def test_load_temperature_sidecar_returns_none_when_missing(tmp_path):
    assert load_temperature_sidecar(tmp_path) is None
