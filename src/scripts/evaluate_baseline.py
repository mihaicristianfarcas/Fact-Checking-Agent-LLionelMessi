#!/usr/bin/env python3
"""FEVER dev-set retrieval baseline and diagnostics.

Measures:
  1. Page-level retrieval recall at configurable k cutoffs.
  2. A simple retrieval-score verdict heuristic.
  3. Calibration (ECE) for that heuristic.
  4. Whether gold pages are present in the current Chroma index at all.

Usage:
    python -m src.scripts.evaluate_baseline
    python -m src.scripts.evaluate_baseline --max-claims 500 --top-k 10
    python -m src.scripts.evaluate_baseline --candidate-k 100 --recall-ks 1,5,10,50,100
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix

from src.config import settings
from src.data_ingestion import EvidenceRetriever
from src.data_ingestion.retriever.fever_title_retriever import default_title_index_path
from src.data_ingestion.retriever.hybrid_retriever import HybridEvidenceRetriever
from src.evaluation.fever_utils import (
    LABELS,
    ChromaPagePresence,
    gold_page_presence_stats,
    load_fever_dev_claims,
    recall_at_k,
    serialize_gold_pages,
)
from src.evaluation.metrics import expected_calibration_error


def predict_verdict(top_score: float, threshold: float) -> str:
    """Heuristic: high confidence -> SUPPORTED, low -> NOT_ENOUGH_INFO."""
    return "SUPPORTED" if top_score >= threshold else "NOT_ENOUGH_INFO"


def parse_recall_ks(value: str) -> list[int]:
    """Parse comma-separated recall cutoffs."""
    ks = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not ks:
        raise ValueError("--recall-ks must include at least one integer")
    return sorted(set(ks))


def build_retriever(args, retrieval_k: int):
    """Build dense, title-only, or hybrid retrieval for diagnostics."""
    dense = EvidenceRetriever(
        index_path=str(settings.get_absolute_path(settings.chroma_persist_dir)),
        collection_name=settings.chroma_collection_name,
        embedding_model=settings.embedding_model,
    )
    if args.retrieval_mode == "dense" and not args.enable_reranker:
        return dense

    return HybridEvidenceRetriever(
        dense_retriever=dense,
        enable_dense_retrieval=args.retrieval_mode in {"dense", "hybrid"},
        enable_title_retrieval=args.retrieval_mode in {"title", "hybrid"},
        enable_reranker=args.enable_reranker,
        candidate_k=max(args.candidate_k or retrieval_k, retrieval_k),
        title_candidate_k=args.title_candidate_k,
        title_candidate_pages=args.title_candidate_pages,
        title_index_path=args.title_index_path,
        reranker_model=args.reranker_model,
    )


def main():
    parser = argparse.ArgumentParser(description="FEVER retrieval evaluation baseline")
    parser.add_argument(
        "--max-claims",
        type=int,
        default=1000,
        help="Number of dev claims to evaluate (0 = full set).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Verdict heuristic top-k display depth.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Candidate depth to retrieve for recall diagnostics.",
    )
    parser.add_argument(
        "--recall-ks",
        type=str,
        default="1,5,10",
        help="Comma-separated Recall@k cutoffs, e.g. 1,5,10,50,100.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Similarity threshold for SUPPORTED vs NOT_ENOUGH_INFO.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write results JSON.",
    )
    parser.add_argument(
        "--trace-output",
        type=str,
        default=None,
        help="Optional path to write per-claim retrieval traces.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["dense", "title", "hybrid"],
        default="dense",
        help="Retrieval source for baseline diagnostics.",
    )
    parser.add_argument(
        "--enable-reranker",
        action="store_true",
        help="Rerank retrieved candidates before computing Recall@k.",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="SentenceTransformers CrossEncoder model for reranking.",
    )
    parser.add_argument(
        "--title-index-path",
        type=str,
        default=str(default_title_index_path()),
        help="Path to data/index/fever_titles.sqlite.",
    )
    parser.add_argument(
        "--title-candidate-pages",
        type=int,
        default=20,
        help="Number of FEVER title-matched pages to inspect.",
    )
    parser.add_argument(
        "--title-candidate-k",
        type=int,
        default=50,
        help="Number of title/page passages to add before reranking.",
    )
    args = parser.parse_args()

    recall_ks = parse_recall_ks(args.recall_ks)
    retrieval_k = max(args.top_k, args.candidate_k or args.top_k, max(recall_ks))

    claims, _ = load_fever_dev_claims(args.max_claims)

    logger.info("Initialising retriever...")
    retriever = build_retriever(args, retrieval_k)

    logger.info(f"Retrieving top-{retrieval_k} passages for each claim...")

    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float] = []
    correct_flags: list[bool] = []
    trace_rows: list[dict] = []

    recall_hits = defaultdict(int)
    n_with_gold = 0

    claim_texts = [c["claim"] for c in claims]
    batch_results = retriever.retrieve_batch(claim_texts, top_k=retrieval_k)

    for claim, results in zip(claims, batch_results):
        gold = claim["label"]
        gold_pages = claim["gold_pages"]

        top_score = results[0].score if results else 0.0
        pred = predict_verdict(top_score, args.threshold)

        y_true.append(gold)
        y_pred.append(pred)
        confidences.append(top_score)
        correct_flags.append(pred == gold)

        if gold_pages:
            n_with_gold += 1
            for k in recall_ks:
                if recall_at_k(results, gold_pages, k):
                    recall_hits[k] += 1

        if args.trace_output:
            trace_rows.append(
                {
                    "claim_id": claim["id"],
                    "claim": claim["claim"],
                    "gold_label": gold,
                    "gold_pages": serialize_gold_pages(gold_pages),
                    "top_score": top_score,
                    "predicted_label": pred,
                    "recall_hits": {
                        f"recall_at_{k}": recall_at_k(results, gold_pages, k)
                        for k in recall_ks
                    },
                    "retrieved": [
                        {
                            "rank": r.rank,
                            "passage_id": r.passage.id,
                            "source": r.passage.source,
                            "score": r.score,
                            "methods": r.passage.metadata.get("retrieval_methods")
                            or r.passage.metadata.get("retrieval_method"),
                            "text_preview": r.passage.text[:160],
                        }
                        for r in results
                    ],
                }
            )

    n = len(y_true)
    accuracy = sum(p == t for p, t in zip(y_pred, y_true)) / n if n else 0.0
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

    chroma_index_dir = settings.get_absolute_path(settings.chroma_persist_dir)
    with ChromaPagePresence.from_index_dir(chroma_index_dir) as presence:
        page_presence = gold_page_presence_stats(claims, presence)

    sep = "=" * 60
    print(f"\n{sep}")
    print("  FEVER Retrieval Baseline - Results")
    print(sep)
    print(f"\n  Claims evaluated : {n:,}")
    print(f"  Top-k requested  : {args.top_k}")
    print(f"  Candidate-k      : {retrieval_k}")
    print(f"  Retrieval mode   : {args.retrieval_mode}")
    print(f"  Reranker enabled : {args.enable_reranker}")
    print(f"  Threshold        : {args.threshold}")

    print(f"\n  Ground-truth distribution:")
    for lbl in LABELS:
        pct = 100 * label_dist[lbl] / n if n else 0
        print(f"    {lbl:<20} {label_dist[lbl]:>6,}  ({pct:.1f}%)")

    print(f"\n-- Retrieval Quality (page-level recall) ---------------------")
    if page_presence["claims_with_gold_pages"]:
        print(
            "  Gold page present in current Chroma index: "
            f"{page_presence['claims_with_gold_page_present']:,}/"
            f"{page_presence['claims_with_gold_pages']:,} "
            f"({page_presence['gold_page_present_rate']:.3f})"
        )
    if n_with_gold:
        for k in recall_ks:
            r = recall_hits[k] / n_with_gold
            print(
                f"  Recall@{k:<3}  {r:.3f}  "
                f"({recall_hits[k]:,}/{n_with_gold:,} claims with gold evidence)"
            )
    else:
        print("  (no gold evidence pages in evaluated subset)")

    print(f"\n-- Verdict Prediction -----------------------------------------")
    print(f"  Accuracy   {accuracy:.3f}")
    print(f"  Macro F1   {macro_f1:.3f}")
    print(f"  ECE        {ece:.3f}  (0 = perfectly calibrated)")
    print()
    print("  Per-class F1:")
    for lbl in LABELS:
        f1 = report[lbl]["f1-score"]
        prec = report[lbl]["precision"]
        rec = report[lbl]["recall"]
        sup = int(report[lbl]["support"])
        print(f"    {lbl:<20}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  (n={sup:,})")

    print()
    print("  Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    header = "  " + "".join(f"{l[:9]:>12}" for l in LABELS)
    print(header)
    for lbl, row in zip(LABELS, cm):
        print(f"  {lbl[:9]:<12}" + "".join(f"{v:>12,}" for v in row))

    print(f"\n{sep}\n")

    results_dict = {
        "n_claims": n,
        "top_k": args.top_k,
        "candidate_k": retrieval_k,
        "retrieval_mode": args.retrieval_mode,
        "index_type": "current_chroma"
        if args.retrieval_mode == "dense"
        else f"current_chroma_plus_fever_title_{args.retrieval_mode}",
        "embedding_model": settings.embedding_model,
        "enable_reranker": args.enable_reranker,
        "reranker_model": args.reranker_model if args.enable_reranker else None,
        "title_index_path": args.title_index_path
        if args.retrieval_mode in {"title", "hybrid"}
        else None,
        "title_candidate_pages": args.title_candidate_pages,
        "title_candidate_k": args.title_candidate_k,
        "threshold": args.threshold,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "ece": ece,
        **page_presence,
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

    out = Path(args.output or "data/processed/baseline_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results_dict, indent=2), encoding="utf-8")
    logger.info(f"Results written to {out}")

    if args.trace_output:
        trace_path = Path(args.trace_output)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(trace_rows, indent=2), encoding="utf-8")
        logger.info(f"Per-claim retrieval trace written to {trace_path}")


if __name__ == "__main__":
    main()
