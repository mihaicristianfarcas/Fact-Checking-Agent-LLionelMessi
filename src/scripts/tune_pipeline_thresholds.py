"""Held-out threshold tuning for the trained-verifier ensemble.

Reads a raw-predictions JSONL emitted by ``evaluate_pipeline --raw-predictions-output``,
splits it deterministically into a tune half and a holdout half, grid-sweeps
verifier + ensemble thresholds on the tune half, picks the best macro-F1
combination, then reports metrics on the untouched holdout half.

This addresses a methodology bug where the previously published 500-claim
result tuned thresholds on the same 500 claims it reported on.

Usage:
    python -m src.scripts.tune_pipeline_thresholds \
        --raw-predictions data/processed/raw_predictions_500.jsonl \
        --output data/processed/tuned_thresholds.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from loguru import logger
from sklearn.metrics import f1_score

from src.claim_processing.verdict_verifier import (
    VerifierPrediction,
    apply_baseline_refute_fallback,
    calibrate_verifier_prediction,
)
from src.evaluation.metrics import LABELS
from src.synthesis.verdict_synthesizer import SynthesisResult


def _stable_split(records: list[dict], split: str, ratio: float = 0.5) -> list[dict]:
    """Deterministic 50/50 split by hashing claim_id."""
    out: list[dict] = []
    for record in records:
        h = int(hashlib.sha256(str(record["claim_id"]).encode()).hexdigest(), 16)
        bucket = (h % 1000) / 1000.0
        if split == "tune" and bucket < ratio:
            out.append(record)
        elif split == "holdout" and bucket >= ratio:
            out.append(record)
    return out


def _resolve_prediction(
    record: dict,
    *,
    supported_threshold: float,
    refuted_threshold: float,
    min_margin: float,
    use_baseline_fallback: bool,
    ensemble_baseline_refute_threshold: float,
    ensemble_verifier_refute_prob_threshold: float,
) -> str:
    """Apply calibration + optional baseline fallback to a raw record."""
    probs = record["verifier_probabilities"]
    label = max(probs, key=probs.get)
    prediction = VerifierPrediction(
        label=label,
        confidence=probs[label],
        probabilities=probs,
    )
    prediction = calibrate_verifier_prediction(
        prediction,
        min_margin=min_margin,
        supported_threshold=supported_threshold,
        refuted_threshold=refuted_threshold,
    )
    if use_baseline_fallback:
        baseline = SynthesisResult(
            original_claim="",
            verdict=record["baseline_verdict"],
            confidence=float(record["baseline_confidence"]),
            explanation="",
            cited_passage_ids=[],
            atomic_verdicts=[],
            all_retrieved_ids=[],
        )
        prediction = apply_baseline_refute_fallback(
            prediction,
            baseline,
            min_baseline_confidence=ensemble_baseline_refute_threshold,
            min_verifier_refute_probability=ensemble_verifier_refute_prob_threshold,
        )
    return prediction.label


def _macro_f1(records: list[dict], **threshold_kwargs) -> tuple[float, float]:
    y_true = [r["gold_label"] for r in records]
    y_pred = [_resolve_prediction(r, **threshold_kwargs) for r in records]
    macro = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / max(1, len(y_true))
    return macro, acc


def _grid_sweep(records: list[dict]) -> dict:
    supported_grid = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    refuted_grid = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    margin_grid = [0.0, 0.05, 0.08, 0.10, 0.15]
    baseline_refute_grid = [0.75, 0.80, 0.82, 0.85, 0.90]
    verifier_refute_prob_grid = [0.20, 0.25, 0.30, 0.35]

    best = {"macro_f1": -1.0, "thresholds": None}
    total = (
        len(supported_grid)
        * len(refuted_grid)
        * len(margin_grid)
        * (1 + len(baseline_refute_grid) * len(verifier_refute_prob_grid))
    )
    logger.info("Sweeping {} threshold combinations", total)

    iterator = itertools.product(supported_grid, refuted_grid, margin_grid)
    for sup_t, ref_t, margin in iterator:
        # Without fallback first.
        macro, acc = _macro_f1(
            records,
            supported_threshold=sup_t,
            refuted_threshold=ref_t,
            min_margin=margin,
            use_baseline_fallback=False,
            ensemble_baseline_refute_threshold=0.0,
            ensemble_verifier_refute_prob_threshold=0.0,
        )
        if macro > best["macro_f1"]:
            best = {
                "macro_f1": macro,
                "accuracy": acc,
                "thresholds": {
                    "supported_threshold": sup_t,
                    "refuted_threshold": ref_t,
                    "min_margin": margin,
                    "use_baseline_fallback": False,
                    "ensemble_baseline_refute_threshold": None,
                    "ensemble_verifier_refute_prob_threshold": None,
                },
            }
        # With fallback.
        for base_t, ver_p in itertools.product(
            baseline_refute_grid, verifier_refute_prob_grid
        ):
            macro, acc = _macro_f1(
                records,
                supported_threshold=sup_t,
                refuted_threshold=ref_t,
                min_margin=margin,
                use_baseline_fallback=True,
                ensemble_baseline_refute_threshold=base_t,
                ensemble_verifier_refute_prob_threshold=ver_p,
            )
            if macro > best["macro_f1"]:
                best = {
                    "macro_f1": macro,
                    "accuracy": acc,
                    "thresholds": {
                        "supported_threshold": sup_t,
                        "refuted_threshold": ref_t,
                        "min_margin": margin,
                        "use_baseline_fallback": True,
                        "ensemble_baseline_refute_threshold": base_t,
                        "ensemble_verifier_refute_prob_threshold": ver_p,
                    },
                }
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-ratio", type=float, default=0.5)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in Path(args.raw_predictions).read_text("utf-8").splitlines()
        if line.strip()
    ]
    logger.info("Loaded {} raw records", len(records))

    tune = _stable_split(records, "tune", args.split_ratio)
    holdout = _stable_split(records, "holdout", args.split_ratio)
    logger.info("Tune={}, Holdout={}", len(tune), len(holdout))

    best = _grid_sweep(tune)
    logger.info("Best on tune: macro_f1={:.4f}", best["macro_f1"])
    logger.info("Thresholds: {}", best["thresholds"])

    holdout_macro, holdout_acc = _macro_f1(holdout, **best["thresholds"])
    logger.info(
        "Holdout: macro_f1={:.4f}, accuracy={:.4f}", holdout_macro, holdout_acc
    )

    payload = {
        "tune_size": len(tune),
        "holdout_size": len(holdout),
        "split_ratio": args.split_ratio,
        "tune_best_macro_f1": best["macro_f1"],
        "tune_best_accuracy": best["accuracy"],
        "holdout_macro_f1": holdout_macro,
        "holdout_accuracy": holdout_acc,
        "thresholds": best["thresholds"],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote {}", out)


if __name__ == "__main__":
    main()
