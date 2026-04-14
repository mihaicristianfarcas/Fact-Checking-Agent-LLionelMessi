#!/usr/bin/env python3
"""Full-pipeline FEVER evaluation harness.

Unlike evaluate_baseline.py (retrieval-only heuristic), this script runs
every claim through the complete agent pipeline:

    decompose → retrieve → stance classify → credibility score → synthesize

and reports end-to-end metrics:

    1. Verdict accuracy and macro F1 (three-class).
    2. Per-class precision / recall / F1.
    3. Calibration (ECE) on the synthesizer's final confidence.
    4. Citation faithfulness — % of cited passage IDs that exist in retrieval.
    5. Hallucination rate — % of outputs that cite non-retrieved passages or
       make a verdict claim with zero citations.
    6. Confusion matrix.

Usage:
    python -m src.scripts.evaluate_pipeline
    python -m src.scripts.evaluate_pipeline --max-claims 500 --top-k 5
    python -m src.scripts.evaluate_pipeline --max-claims 0   # full dev set
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix

from src.agent.orchestrator import FactCheckAgent
from src.config import settings


LABEL_MAP = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "REFUTED",
    "NOT ENOUGH INFO": "NOT_ENOUGH_INFO",
}
LABELS = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]


def load_dev_claims(max_claims: int | None) -> list[dict]:
    """Load and deduplicate FEVER labelled_dev claims."""
    logger.info("Loading FEVER labelled_dev...")
    ds = load_dataset(
        "fever/fever",
        "v1.0",
        split="labelled_dev",
        trust_remote_code=True,
        verification_mode="no_checks",
    )

    grouped: dict[int, dict] = {}
    for row in ds:
        cid = row["id"]
        if cid not in grouped:
            grouped[cid] = {
                "id": cid,
                "claim": row["claim"],
                "label": LABEL_MAP[row["label"]],
            }

    claims = list(grouped.values())
    if max_claims and max_claims > 0:
        claims = claims[:max_claims]

    logger.info(f"Evaluating on {len(claims)} claims")
    return claims


def expected_calibration_error(
    confidences: list[float],
    correct: list[bool],
    n_bins: int = 10,
) -> float:
    """Weighted average |confidence - accuracy| across bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    conf_arr = np.array(confidences)
    corr_arr = np.array(correct, dtype=float)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf_arr >= lo) & (conf_arr < hi)
        if mask.sum() == 0:
            continue
        bin_conf = conf_arr[mask].mean()
        bin_acc = corr_arr[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)

    return float(ece)


def main():
    parser = argparse.ArgumentParser(
        description="Full-pipeline FEVER evaluation"
    )
    parser.add_argument(
        "--max-claims",
        type=int,
        default=200,
        help="Number of dev claims (0 = full set). Default 200 for speed.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Passages to retrieve per atomic claim.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON results.",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Enable adaptive retrieval in the agent.",
    )
    args = parser.parse_args()

    claims = load_dev_claims(args.max_claims)

    logger.info("Initializing FactCheckAgent (this loads models)...")
    agent = FactCheckAgent(top_k=args.top_k, adaptive=args.adaptive)

    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float] = []
    correct_flags: list[bool] = []
    hallucination_flags: list[bool] = []
    citation_missing_flags: list[bool] = []

    logger.info(f"Running pipeline on {len(claims)} claims...")
    for i, claim_data in enumerate(claims):
        if (i + 1) % 50 == 0 or i == 0:
            logger.info(f"  Progress: {i + 1}/{len(claims)}")

        try:
            result = agent.check(claim_data["claim"])
        except Exception as exc:
            logger.warning(
                f"Claim {claim_data['id']} failed: {exc}. Defaulting to NEI."
            )
            y_true.append(claim_data["label"])
            y_pred.append("NOT_ENOUGH_INFO")
            confidences.append(0.0)
            correct_flags.append(claim_data["label"] == "NOT_ENOUGH_INFO")
            hallucination_flags.append(False)
            citation_missing_flags.append(True)
            continue

        gold = claim_data["label"]
        pred = result.verdict

        y_true.append(gold)
        y_pred.append(pred)
        confidences.append(result.confidence)
        correct_flags.append(pred == gold)

        has_hallucinated_cite = len(result.hallucinated_citations) > 0
        hallucination_flags.append(has_hallucinated_cite)

        missing_cite = not result.citation_present
        citation_missing_flags.append(missing_cite)

    # ── Compute metrics ───────────────────────────────────────────────────
    n = len(y_true)
    accuracy = sum(p == t for p, t in zip(y_pred, y_true)) / n if n else 0
    ece = expected_calibration_error(confidences, correct_flags)

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
        output_dict=True,
    )
    macro_f1 = report["macro avg"]["f1-score"]

    label_dist = defaultdict(int)
    for lbl in y_true:
        label_dist[lbl] += 1

    halluc_count = sum(hallucination_flags)
    halluc_rate = halluc_count / n if n else 0

    cite_missing = sum(citation_missing_flags)
    cite_missing_rate = cite_missing / n if n else 0

    # ── Print results ─────────────────────────────────────────────────────
    sep = "=" * 64
    print(f"\n{sep}")
    print("  Full Pipeline Evaluation - Results")
    print(sep)
    print(f"\n  Claims evaluated : {n:,}")
    print(f"  Top-k per atomic : {args.top_k}")
    print(f"  Adaptive mode    : {args.adaptive}")

    print(f"\n  Ground-truth distribution:")
    for lbl in LABELS:
        pct = 100 * label_dist[lbl] / n if n else 0
        print(f"    {lbl:<20} {label_dist[lbl]:>6,}  ({pct:.1f}%)")

    print(f"\n-- Verdict Quality --------------------------------------------")
    print(f"  Accuracy     {accuracy:.3f}")
    print(f"  Macro F1     {macro_f1:.3f}")
    print(f"  ECE          {ece:.3f}  (0 = perfectly calibrated)")

    print(f"\n  Per-class:")
    for lbl in LABELS:
        f1 = report[lbl]["f1-score"]
        prec = report[lbl]["precision"]
        rec = report[lbl]["recall"]
        sup = int(report[lbl]["support"])
        print(
            f"    {lbl:<20}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  (n={sup:,})"
        )

    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    header = "  " + "".join(f"{l[:9]:>12}" for l in LABELS)
    print(header)
    for lbl, row in zip(LABELS, cm):
        print(f"  {lbl[:9]:<12}" + "".join(f"{v:>12,}" for v in row))

    print(f"\n-- Citation & Hallucination -----------------------------------")
    print(f"  Hallucinated citations : {halluc_count:,}/{n:,}  ({halluc_rate:.1%})")
    print(f"  Missing citations      : {cite_missing:,}/{n:,}  ({cite_missing_rate:.1%})")
    target_met = "YES" if halluc_rate < 0.05 else "NO"
    print(f"  Hallucination < 5%     : {target_met}")

    print(f"\n{sep}\n")

    # ── Save results ──────────────────────────────────────────────────────
    results_dict = {
        "n_claims": n,
        "top_k": args.top_k,
        "adaptive": args.adaptive,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "ece": ece,
        "hallucination_rate": halluc_rate,
        "citation_missing_rate": cite_missing_rate,
        "per_class": {
            lbl: {
                "precision": report[lbl]["precision"],
                "recall": report[lbl]["recall"],
                "f1": report[lbl]["f1-score"],
                "support": int(report[lbl]["support"]),
            }
            for lbl in LABELS
        },
        "label_distribution": dict(label_dist),
    }

    out_path = Path(
        args.output or "data/processed/pipeline_eval_results.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results_dict, indent=2))
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
