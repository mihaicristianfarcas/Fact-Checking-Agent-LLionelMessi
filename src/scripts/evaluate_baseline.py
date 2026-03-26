#!/usr/bin/env python3
"""FEVER dev-set evaluation baseline for the Evidence Retriever.

Measures three things:
  1. Retrieval recall  — does the gold Wikipedia page appear in top-k results?
  2. Verdict accuracy / macro F1 — using a confidence-threshold heuristic.
  3. Calibration (ECE) — does the top-1 similarity score reflect actual accuracy?

Limitation: without a Stance Classifier the retriever alone cannot distinguish
SUPPORTED from REFUTED (both require finding evidence). This script therefore
predicts only SUPPORTED or NOT_ENOUGH_INFO.  REFUTED claims will be mis-
classified as SUPPORTED whenever the retriever finds a highly-similar passage.
This is the expected floor before Person B's Stance Classifier is integrated.

Usage:
    python -m src.scripts.evaluate_baseline
    python -m src.scripts.evaluate_baseline --max-claims 2000 --top-k 10
    python -m src.scripts.evaluate_baseline --max-claims 0          # full dev set
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix

from src.config import settings
from src.data_ingestion import EvidenceRetriever


# ── helpers ──────────────────────────────────────────────────────────────────

LABEL_MAP = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "REFUTED",
    "NOT ENOUGH INFO": "NOT_ENOUGH_INFO",
}
LABELS = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]


def load_dev_claims(max_claims: int | None) -> list[dict]:
    """Load and group labelled_dev rows by claim id.

    Returns a list of dicts:
        {id, claim, label, gold_pages: set[str]}
    where gold_pages is the set of Wikipedia page titles cited as evidence.
    """
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
                "gold_pages": set(),
            }
        if row["evidence_wiki_url"]:
            grouped[cid]["gold_pages"].add(row["evidence_wiki_url"])

    claims = list(grouped.values())
    if max_claims and max_claims > 0:
        claims = claims[:max_claims]

    logger.info(f"Evaluating on {len(claims)} claims")
    return claims


def predict_verdict(top_score: float, threshold: float) -> str:
    """Heuristic: high confidence → SUPPORTED, low → NOT_ENOUGH_INFO.

    Cannot predict REFUTED without a stance classifier — that is the explicit
    gap this baseline is designed to expose.
    """
    return "SUPPORTED" if top_score >= threshold else "NOT_ENOUGH_INFO"


def expected_calibration_error(
    confidences: list[float],
    correct: list[bool],
    n_bins: int = 10,
) -> float:
    """Compute ECE: weighted average |confidence - accuracy| across bins."""
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


def recall_at_k(results, gold_pages: set[str], k: int) -> bool:
    """True if any of the top-k passages comes from a gold Wikipedia page."""
    top_k = results[:k]
    return any(r.passage.source in gold_pages for r in top_k)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FEVER retrieval evaluation baseline")
    parser.add_argument(
        "--max-claims",
        type=int,
        default=1000,
        help="Number of dev claims to evaluate (0 = full set, ~20k, takes ~30 min)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of passages to retrieve per claim",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Cosine similarity threshold for SUPPORTED vs NOT_ENOUGH_INFO",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write results JSON",
    )
    args = parser.parse_args()

    # ── load data ──────────────────────────────────────────────────────────
    claims = load_dev_claims(args.max_claims)

    # ── init retriever ─────────────────────────────────────────────────────
    logger.info("Initialising EvidenceRetriever...")
    retriever = EvidenceRetriever(
        index_path=str(settings.get_absolute_path(settings.chroma_persist_dir)),
        collection_name=settings.chroma_collection_name,
        embedding_model=settings.embedding_model,
    )

    # ── evaluate ───────────────────────────────────────────────────────────
    logger.info(f"Retrieving top-{args.top_k} passages for each claim...")

    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float] = []
    correct_flags: list[bool] = []

    recall_hits = defaultdict(int)   # k -> count of hits
    recall_ks = [1, 5, 10]
    n_with_gold = 0                  # claims that have at least one gold page

    claim_texts = [c["claim"] for c in claims]
    batch_results = retriever.retrieve_batch(claim_texts, top_k=args.top_k)

    for claim, results in zip(claims, batch_results):
        gold = claim["label"]
        gold_pages = claim["gold_pages"]

        top_score = results[0].score if results else 0.0
        pred = predict_verdict(top_score, args.threshold)

        y_true.append(gold)
        y_pred.append(pred)
        confidences.append(top_score)
        correct_flags.append(pred == gold)

        # retrieval recall (only meaningful for claims with cited evidence)
        if gold_pages:
            n_with_gold += 1
            for k in recall_ks:
                if recall_at_k(results, gold_pages, k):
                    recall_hits[k] += 1

    # ── compute metrics ────────────────────────────────────────────────────
    n = len(y_true)
    accuracy = sum(p == t for p, t in zip(y_pred, y_true)) / n
    ece = expected_calibration_error(confidences, correct_flags)

    report = classification_report(
        y_true, y_pred,
        labels=LABELS,
        zero_division=0,
        output_dict=True,
    )
    macro_f1 = report["macro avg"]["f1-score"]

    label_dist = defaultdict(int)
    for lbl in y_true:
        label_dist[lbl] += 1

    # ── print results ──────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print("  FEVER Retrieval Baseline — Results")
    print(sep)
    print(f"\n  Claims evaluated : {n:,}")
    print(f"  Top-k retrieved  : {args.top_k}")
    print(f"  Threshold        : {args.threshold}")

    print(f"\n  Ground-truth distribution:")
    for lbl in LABELS:
        pct = 100 * label_dist[lbl] / n
        print(f"    {lbl:<20} {label_dist[lbl]:>6,}  ({pct:.1f}%)")

    print(f"\n── Retrieval Quality (page-level recall) ──────────────────")
    if n_with_gold:
        for k in recall_ks:
            r = recall_hits[k] / n_with_gold
            print(f"  Recall@{k:<3}  {r:.3f}  ({recall_hits[k]:,}/{n_with_gold:,} claims with gold evidence)")
    else:
        print("  (no gold evidence pages in evaluated subset)")

    print(f"\n── Verdict Prediction ─────────────────────────────────────")
    print(f"  Accuracy   {accuracy:.3f}")
    print(f"  Macro F1   {macro_f1:.3f}")
    print(f"  ECE        {ece:.3f}  (0 = perfectly calibrated)")
    print()
    print("  Per-class F1:")
    for lbl in LABELS:
        f1  = report[lbl]["f1-score"]
        prec = report[lbl]["precision"]
        rec  = report[lbl]["recall"]
        sup  = int(report[lbl]["support"])
        print(f"    {lbl:<20}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  (n={sup:,})")

    print()
    print("  Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    header = "  " + "".join(f"{l[:9]:>12}" for l in LABELS)
    print(header)
    for lbl, row in zip(LABELS, cm):
        print(f"  {lbl[:9]:<12}" + "".join(f"{v:>12,}" for v in row))

    print(f"\n── Interpretation ─────────────────────────────────────────")
    print("  This baseline uses retrieval confidence alone — no Stance Classifier.")
    print("  REFUTED claims are systematically mis-classified as SUPPORTED")
    print("  whenever the retriever finds a passage (regardless of stance).")
    print("  Recall@k measures how often the correct Wikipedia page appears")
    print("  in the top-k results; this is what Person B's Stance Classifier")
    print("  will operate on.")
    print(f"\n{sep}\n")

    # ── optional JSON output ───────────────────────────────────────────────
    results_dict = {
        "n_claims": n,
        "top_k": args.top_k,
        "threshold": args.threshold,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "ece": ece,
        "retrieval_recall": {
            f"recall_at_{k}": recall_hits[k] / n_with_gold if n_with_gold else None
            for k in recall_ks
        },
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

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results_dict, indent=2))
        logger.info(f"Results written to {out}")
    else:
        # Always save alongside the processed data as a default artifact
        default_out = Path("data/processed/baseline_results.json")
        default_out.parent.mkdir(parents=True, exist_ok=True)
        default_out.write_text(json.dumps(results_dict, indent=2))
        logger.info(f"Results saved to {default_out}")


if __name__ == "__main__":
    main()
