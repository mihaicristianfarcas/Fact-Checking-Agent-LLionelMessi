#!/usr/bin/env python3
"""Build retrieved-evidence FEVER verifier datasets from FEVER train.

The output rows train a claim-level classifier:

    claim + retrieved evidence passages -> SUPPORTED / REFUTED / NOT_ENOUGH_INFO

Important: this script uses the FEVER train split only. It intentionally does
not read labelled_dev, which remains the evaluation set.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

from datasets import load_dataset
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.agent.orchestrator import combine_same_source_evidence
from src.claim_processing.verdict_verifier import (
    build_evidence_text,
    build_verifier_input,
)
from src.data_ingestion.retriever.fever_title_retriever import default_title_index_path
from src.data_ingestion.retriever.hybrid_retriever import HybridEvidenceRetriever
from src.evaluation.fever_utils import LABEL_MAP, LABELS


def load_fever_train_claims(
    *,
    max_claims: int = 0,
    seed: int = 42,
    shuffle: bool = True,
) -> list[dict[str, Any]]:
    """Load unique FEVER train claims with page-level gold evidence."""
    logger.info("Loading FEVER train split...")
    ds = load_dataset(
        "fever/fever",
        "v1.0",
        split="train",
        trust_remote_code=True,
        verification_mode="no_checks",
    )

    grouped: OrderedDict[int, dict[str, Any]] = OrderedDict()
    for row in ds:
        cid = int(row["id"])
        if cid not in grouped:
            grouped[cid] = {
                "id": cid,
                "claim": row["claim"],
                "label": LABEL_MAP[row["label"]],
                "gold_pages": set(),
            }

        evidence_page = row.get("evidence_wiki_url")
        if isinstance(evidence_page, list):
            grouped[cid]["gold_pages"].update(page for page in evidence_page if page)
        elif evidence_page:
            grouped[cid]["gold_pages"].add(evidence_page)

    claims = list(grouped.values())
    if shuffle:
        random.Random(seed).shuffle(claims)

    if max_claims and max_claims > 0:
        claims = claims[:max_claims]

    logger.info(f"Loaded {len(claims):,} unique FEVER train claims")
    return claims


def build_retriever(args: argparse.Namespace) -> HybridEvidenceRetriever:
    """Construct the same hybrid retriever family used by pipeline eval."""
    return HybridEvidenceRetriever(
        enable_dense_retrieval=not args.disable_dense_retrieval,
        enable_title_retrieval=not args.disable_title_retrieval,
        enable_reranker=args.enable_reranker,
        candidate_k=max(args.candidate_k, args.top_k),
        title_candidate_k=args.title_candidate_k,
        title_candidate_pages=args.title_candidate_pages,
        title_index_path=args.title_index_path,
        reranker_model=args.reranker_model,
    )


def build_rows(
    claims: list[dict[str, Any]],
    retriever: HybridEvidenceRetriever,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], Counter]:
    """Retrieve evidence and convert claims into verifier JSONL rows."""
    rows: list[dict[str, Any]] = []
    stats: Counter = Counter()

    for start in tqdm(
        range(0, len(claims), args.retrieval_batch_size),
        desc="Retrieving FEVER train evidence",
    ):
        chunk = claims[start : start + args.retrieval_batch_size]
        retrieval_batches = retriever.retrieve_batch(
            [claim["claim"] for claim in chunk],
            top_k=args.top_k,
        )

        for claim_data, retrievals in zip(chunk, retrieval_batches):
            label = claim_data["label"]
            gold_pages = set(claim_data["gold_pages"])
            has_gold_page = bool(gold_pages) and any(
                result.passage.source in gold_pages for result in retrievals
            )

            if not retrievals and not args.keep_empty_evidence:
                stats["skipped_empty_evidence"] += 1
                continue

            if (
                label in {"SUPPORTED", "REFUTED"}
                and not has_gold_page
                and not args.keep_positive_without_gold_page
            ):
                stats["skipped_positive_missing_gold_page"] += 1
                continue

            input_retrievals = retrievals
            if args.combine_same_source_evidence:
                input_retrievals = combine_same_source_evidence(
                    retrievals,
                    max_passages_per_source=args.same_source_max_passages,
                )

            evidence_text = build_evidence_text(
                input_retrievals,
                max_passages=args.max_passages,
            )
            input_text = build_verifier_input(
                claim_data["claim"],
                input_retrievals,
                max_passages=args.max_passages,
            )

            rows.append(
                {
                    "claim_id": str(claim_data["id"]),
                    "claim": claim_data["claim"],
                    "evidence_text": evidence_text,
                    "label": label,
                    "input_text": input_text,
                    "gold_pages": sorted(gold_pages),
                    "retrieved_gold_page": has_gold_page,
                    "retrieved_passages": [
                        serialize_retrieval(result) for result in input_retrievals
                    ],
                }
            )
            stats[f"kept_{label}"] += 1

    return rows, stats


def serialize_retrieval(result) -> dict[str, Any]:
    """Compact JSON-friendly retrieval record for audit/debugging."""
    metadata = result.passage.metadata or {}
    return {
        "id": result.passage.id,
        "rank": result.rank,
        "score": result.score,
        "source": result.passage.source,
        "dataset": result.passage.dataset,
        "text": result.passage.text,
        "retrieval_methods": metadata.get("retrieval_methods")
        or metadata.get("retrieval_method"),
        "rerank_score": metadata.get("rerank_score"),
        "source_relevance": metadata.get("source_relevance"),
    }


def balance_rows_by_label(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Downsample classes to the smallest label count."""
    by_label = {label: [] for label in LABELS}
    for row in rows:
        by_label[row["label"]].append(row)

    min_count = min((len(items) for items in by_label.values() if items), default=0)
    if min_count == 0:
        return rows

    rng = random.Random(seed)
    balanced: list[dict[str, Any]] = []
    for label in LABELS:
        items = by_label[label]
        rng.shuffle(items)
        balanced.extend(items[:min_count])
    rng.shuffle(balanced)
    return balanced


def split_rows(
    rows: list[dict[str, Any]],
    *,
    dev_size: int,
    dev_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create a deterministic train/dev split from FEVER train rows only."""
    if not rows or (dev_size <= 0 and dev_fraction <= 0):
        return rows, []

    labels = [row["label"] for row in rows]
    test_size: int | float = dev_size if dev_size > 0 else dev_fraction
    try:
        train_rows, dev_rows = train_test_split(
            rows,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=labels,
        )
    except ValueError:
        logger.warning("Stratified split failed; falling back to random split")
        rng = random.Random(seed)
        shuffled = rows[:]
        rng.shuffle(shuffled)
        n_dev = dev_size if dev_size > 0 else round(len(rows) * dev_fraction)
        dev_rows = shuffled[:n_dev]
        train_rows = shuffled[n_dev:]

    return list(train_rows), list(dev_rows)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(rows):,} rows to {output_path}")


def log_distribution(name: str, rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["label"] for row in rows)
    logger.info(f"{name} label distribution: {dict(counts)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build retrieved-evidence FEVER verifier JSONL datasets"
    )
    parser.add_argument(
        "--max-claims",
        type=int,
        default=0,
        help="Max unique FEVER train claims to scan after deterministic shuffle (0 = all).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Training JSONL output path.",
    )
    parser.add_argument(
        "--dev-output",
        type=str,
        default=None,
        help="Optional dev JSONL output path split from FEVER train.",
    )
    parser.add_argument(
        "--dev-size",
        type=int,
        default=0,
        help="Number of rows to put in dev split when --dev-output is set.",
    )
    parser.add_argument(
        "--dev-fraction",
        type=float,
        default=0.0,
        help="Fraction of rows to put in dev split when --dev-output is set.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Use FEVER train order before --max-claims instead of deterministic shuffle.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument(
        "--retrieval-batch-size",
        type=int,
        default=256,
        help="Number of claims per retrieval batch.",
    )
    parser.add_argument(
        "--disable-dense-retrieval",
        action="store_true",
        help="Use title retrieval only.",
    )
    parser.add_argument(
        "--disable-title-retrieval",
        action="store_true",
        help="Use dense retrieval only.",
    )
    parser.add_argument(
        "--title-index-path",
        type=str,
        default=str(default_title_index_path()),
        help="Path to data/index/fever_titles.sqlite.",
    )
    parser.add_argument("--title-candidate-pages", type=int, default=20)
    parser.add_argument("--title-candidate-k", type=int, default=50)
    parser.add_argument(
        "--enable-reranker",
        action="store_true",
        help="Rerank dense/title candidates with the cross-encoder reranker.",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    parser.add_argument(
        "--combine-same-source-evidence",
        action="store_true",
        help="Match eval's same-source evidence-window option.",
    )
    parser.add_argument("--same-source-max-passages", type=int, default=3)
    parser.add_argument(
        "--max-passages",
        type=int,
        default=None,
        help="Max retrieved passages to include in the verifier input.",
    )
    parser.add_argument(
        "--keep-positive-without-gold-page",
        action="store_true",
        help="Keep SUPPORT/REFUTE rows even when no gold page was retrieved.",
    )
    parser.add_argument(
        "--keep-empty-evidence",
        action="store_true",
        help="Keep rows with no retrieved passages.",
    )
    parser.add_argument(
        "--balance-labels",
        action="store_true",
        help="Downsample labels to the smallest kept class before splitting.",
    )
    args = parser.parse_args()

    if args.disable_dense_retrieval and args.disable_title_retrieval:
        raise ValueError("At least one retrieval source must be enabled.")

    claims = load_fever_train_claims(
        max_claims=args.max_claims,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )
    retriever = build_retriever(args)
    rows, stats = build_rows(claims, retriever, args)
    logger.info(f"Build stats: {dict(stats)}")

    if args.balance_labels:
        before = len(rows)
        rows = balance_rows_by_label(rows, args.seed)
        logger.info(f"Balanced rows from {before:,} to {len(rows):,}")

    if args.dev_output:
        dev_size = args.dev_size
        dev_fraction = args.dev_fraction
        if dev_size <= 0 and dev_fraction <= 0:
            dev_fraction = 0.10
        train_rows, dev_rows = split_rows(
            rows,
            dev_size=dev_size,
            dev_fraction=dev_fraction,
            seed=args.seed,
        )
        log_distribution("train", train_rows)
        log_distribution("dev", dev_rows)
        write_jsonl(args.output, train_rows)
        write_jsonl(args.dev_output, dev_rows)
    else:
        log_distribution("output", rows)
        write_jsonl(args.output, rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
