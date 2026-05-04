#!/usr/bin/env python3
"""Evaluate saved fact-checking predictions offline.

Expected JSONL row shape:
    {
      "claim_id": "dev_1",
      "gold_verdict": "SUPPORTED",
      "predicted_verdict": "SUPPORTED",
      "confidence": 0.82,
      "cited_passage_ids": ["p1"],
      "retrieved_passage_ids": ["p1", "p2"]
    }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.metrics import LABELS, evaluate_records, load_prediction_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate verdict quality and citation faithfulness."
    )
    parser.add_argument("predictions", help="JSONL or JSON file of prediction rows.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the full metrics JSON.",
    )
    parser.add_argument(
        "--hallucination-target",
        type=float,
        default=0.05,
        help="Target max hallucination rate. Default: 0.05.",
    )
    args = parser.parse_args()

    records = load_prediction_records(args.predictions)
    metrics = evaluate_records(
        records,
        hallucination_target=args.hallucination_target,
    )

    print("\nOffline Prediction Evaluation")
    print("=" * 64)
    print(f"Claims evaluated      : {metrics['n_claims']}")
    print(f"Accuracy              : {metrics['accuracy']:.3f}")
    print(f"Macro F1              : {metrics['macro_f1']:.3f}")
    print(f"ECE                   : {metrics['ece']:.3f}")
    print()
    print("Per-class metrics:")
    for label in LABELS:
        row = metrics["per_class"][label]
        print(
            f"  {label:<20} P={row['precision']:.3f} "
            f"R={row['recall']:.3f} F1={row['f1']:.3f} n={row['support']}"
        )
    print()
    print("Citation quality:")
    print(
        "  Hallucinated citations : "
        f"{metrics['hallucinated_citation_count']}/{metrics['n_claims']} "
        f"({metrics['hallucination_rate']:.1%})"
    )
    print(
        "  Missing citations      : "
        f"{metrics['missing_citation_count']}/{metrics['n_claims']} "
        f"({metrics['missing_citation_rate']:.1%})"
    )
    print(
        "  Citation violations    : "
        f"{metrics['citation_violation_count']}/{metrics['n_claims']} "
        f"({metrics['citation_violation_rate']:.1%})"
    )
    print(
        "  Hallucination target   : "
        f"{'YES' if metrics['hallucination_target_met'] else 'NO'}"
    )
    print("=" * 64)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
