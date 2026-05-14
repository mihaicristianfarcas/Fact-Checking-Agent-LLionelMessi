#!/usr/bin/env python3
"""Full-pipeline FEVER evaluation harness.

Runs:
    decompose -> retrieve -> stance classify -> credibility score -> synthesize

and reports verdict quality, calibration, citation faithfulness, and optional
wrong-prediction traces for retrieval/stance/synthesis diagnosis.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from src.agent.orchestrator import FactCheckAgent, PipelineTrace
from src.claim_processing.decomposer import AtomicClaim, DecompositionResult
from src.claim_processing.stance_classifier import (
    PassageStance,
    StanceClassifier,
    StanceLabel,
    StanceResult,
)
from src.config import settings
from src.data_ingestion.retriever.fever_title_retriever import default_title_index_path
from src.data_ingestion.retriever.hybrid_retriever import HybridEvidenceRetriever
from src.evaluation.fever_utils import (
    LABELS,
    ChromaPagePresence,
    gold_page_presence_stats,
    load_fever_dev_claims,
    load_train_claim_texts,
    serialize_gold_pages,
)
from src.evaluation.metrics import expected_calibration_error
from src.scoring.credibility_scorer import CredibilityScorer
from src.synthesis.verdict_synthesizer import VerdictSynthesizer


class PassthroughDecomposer:
    """Eval helper that treats each FEVER claim as one atomic claim."""

    def decompose(self, claim: str) -> DecompositionResult:
        return DecompositionResult(
            original_claim=claim,
            atomic_claims=[
                AtomicClaim(
                    text=claim,
                    source_claim=claim,
                    claim_index=0,
                    extraction_method="passthrough",
                )
            ],
            was_compound=False,
            model_used="passthrough",
            latency_ms=0.0,
        )


class NeutralVerifierStanceClassifier:
    """Cheap stance stub used when a trained claim-level verifier is enabled."""

    model_name = "trained-verifier-neutral-stub"

    def classify(self, claim: str, retrievals: list) -> StanceResult:
        passage_stances = [
            PassageStance(
                passage_id=r.passage.id,
                passage_text=r.passage.text,
                passage_source=r.passage.source,
                passage_dataset=r.passage.dataset,
                retrieval_score=r.score,
                retrieval_rank=r.rank,
                stance=StanceLabel.NEUTRAL,
                confidence=1.0,
                raw_scores={
                    StanceLabel.SUPPORTING.value: 0.0,
                    StanceLabel.REFUTING.value: 0.0,
                    StanceLabel.NEUTRAL.value: 1.0,
                },
                passage_metadata=dict(getattr(r.passage, "metadata", {}) or {}),
            )
            for r in retrievals
        ]
        return StanceResult(
            claim_text=claim,
            passage_stances=passage_stances,
            aggregate_label=StanceLabel.NEUTRAL,
            aggregate_score=0.0,
            supporting_count=0,
            refuting_count=0,
            neutral_count=len(passage_stances),
            latency_ms=0.0,
            model_name=self.model_name,
        )


def build_agent(args) -> FactCheckAgent:
    """Construct the pipeline with optional retrieval and threshold overrides."""
    decomposer = PassthroughDecomposer() if args.skip_decomposition else None

    retriever = None
    if args.enable_title_retrieval or args.enable_reranker:
        retriever = HybridEvidenceRetriever(
            enable_dense_retrieval=True,
            enable_title_retrieval=args.enable_title_retrieval,
            enable_reranker=args.enable_reranker,
            candidate_k=max(args.candidate_k, args.top_k),
            title_candidate_k=args.title_candidate_k,
            title_candidate_pages=args.title_candidate_pages,
            title_index_path=args.title_index_path,
            reranker_model=args.reranker_model,
        )

    stance_classifier = None
    if args.use_trained_verifier:
        stance_classifier = NeutralVerifierStanceClassifier()
    elif (
        args.stance_confidence_threshold is not None
        or args.include_source_title_in_stance
    ):
        stance_classifier = StanceClassifier(
            confidence_threshold=args.stance_confidence_threshold
            if args.stance_confidence_threshold is not None
            else settings.stance_confidence_threshold,
            include_source_title_in_premise=args.include_source_title_in_stance,
        )

    credibility_scorer = CredibilityScorer(
        use_source_relevance=not args.disable_source_relevance
    )

    synthesizer = None
    if (
        args.nei_confidence_floor is not None
        or args.decisive_dominance_floor is not None
        or args.disable_refute_overrides
        or args.min_refuting_passages != 1
        or args.single_refute_confidence_floor > 0
        or args.refute_source_relevance_floor is not None
        or args.refute_conflict_margin is not None
    ):
        synthesizer = VerdictSynthesizer(
            credibility_scorer=credibility_scorer,
            nei_confidence_floor=args.nei_confidence_floor
            if args.nei_confidence_floor is not None
            else 0.45,
            refute_overrides=not args.disable_refute_overrides,
            decisive_dominance_floor=args.decisive_dominance_floor
            if args.decisive_dominance_floor is not None
            else 0.60,
            min_refuting_passages=args.min_refuting_passages,
            single_refute_confidence_floor=args.single_refute_confidence_floor,
            refute_source_relevance_floor=args.refute_source_relevance_floor
            if args.refute_source_relevance_floor is not None
            else 0.55,
            refute_conflict_margin=args.refute_conflict_margin
            if args.refute_conflict_margin is not None
            else 0.15,
        )

    return FactCheckAgent(
        decomposer=decomposer,
        retriever=retriever,
        stance_classifier=stance_classifier,
        credibility_scorer=credibility_scorer,
        synthesizer=synthesizer,
        top_k=args.top_k,
        adaptive=args.adaptive,
        combine_same_source_evidence=args.combine_same_source_evidence,
        same_source_max_passages=args.same_source_max_passages,
    )


def summarize_retrievals(trace: PipelineTrace) -> list[dict[str, Any]]:
    """Flatten trace retrievals into JSON-friendly rows."""
    rows: list[dict[str, Any]] = []
    for atomic_claim, results in trace.retrievals.items():
        for r in results:
            metadata = r.passage.metadata
            rows.append(
                {
                    "atomic_claim": atomic_claim,
                    "rank": r.rank,
                    "passage_id": r.passage.id,
                    "source": r.passage.source,
                    "dataset": r.passage.dataset,
                    "score": r.score,
                    "retrieval_methods": metadata.get("retrieval_methods")
                    or metadata.get("retrieval_method"),
                    "dense_score": metadata.get("dense_score"),
                    "title_score": metadata.get("title_score"),
                    "title_match_type": metadata.get("title_match_type"),
                    "rerank_score": metadata.get("rerank_score"),
                    "source_relevance": metadata.get("source_relevance"),
                    "source_relevance_reason": metadata.get(
                        "source_relevance_reason"
                    ),
                    "text_preview": r.passage.text[:240],
                }
            )
    return rows


def summarize_candidates(trace: PipelineTrace, retriever) -> list[dict[str, Any]]:
    """Collect hybrid pre-rerank candidates when the retriever exposes them."""
    if retriever is None or not hasattr(retriever, "get_last_debug"):
        return []

    rows: list[dict[str, Any]] = []
    for atomic_claim in trace.retrievals:
        debug = retriever.get_last_debug(atomic_claim)
        if not debug:
            continue
        for candidate in debug.candidates:
            row = dict(candidate)
            row["atomic_claim"] = atomic_claim
            rows.append(row)
    return rows


def summarize_stances(trace: PipelineTrace) -> list[dict[str, Any]]:
    """Flatten stance classifier outputs into JSON-friendly rows."""
    rows: list[dict[str, Any]] = []
    for atomic_claim, stance_result in trace.stance_results.items():
        for ps in stance_result.passage_stances:
            rows.append(
                {
                    "atomic_claim": atomic_claim,
                    "passage_id": ps.passage_id,
                    "source": ps.passage_source,
                    "rank": ps.retrieval_rank,
                    "retrieval_score": ps.retrieval_score,
                    "stance": ps.stance.value,
                    "confidence": ps.confidence,
                    "raw_scores": ps.raw_scores,
                    "source_relevance": ps.passage_metadata.get("source_relevance"),
                    "source_relevance_reason": ps.passage_metadata.get(
                        "source_relevance_reason"
                    ),
                }
            )
    return rows


def classify_failure_category(
    *,
    gold_label: str,
    predicted_label: str,
    gold_pages: set[str],
    gold_page_present_in_chroma: bool,
    retrieval_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    stance_rows: list[dict[str, Any]],
) -> str:
    """Classify a wrong prediction into a high-level failure mode."""
    if predicted_label == gold_label:
        return "correct"

    final_has_gold = any(row["source"] in gold_pages for row in retrieval_rows)
    candidate_has_gold = any(row["source"] in gold_pages for row in candidate_rows)

    if gold_pages and not final_has_gold:
        if candidate_has_gold:
            return "rerank_dropped_gold_page"
        if not gold_page_present_in_chroma:
            return "gold_page_absent_from_current_index"
        return "candidate_missing"

    expected_stance = {
        "SUPPORTED": "SUPPORTING",
        "REFUTED": "REFUTING",
    }.get(gold_label)
    if expected_stance:
        has_expected_stance = any(
            row["source"] in gold_pages and row["stance"] == expected_stance
            for row in stance_rows
        )
        return "synthesis_wrong" if has_expected_stance else "stance_wrong"

    return "stance_wrong"


def make_error_trace(
    claim_data: dict,
    trace: PipelineTrace,
    retriever,
    page_presence: ChromaPagePresence,
) -> dict[str, Any]:
    """Build one wrong-prediction trace row."""
    result = trace.synthesis
    gold_pages = claim_data["gold_pages"]
    retrieval_rows = summarize_retrievals(trace)
    candidate_rows = summarize_candidates(trace, retriever)
    stance_rows = summarize_stances(trace)
    gold_present = page_presence.any_present(gold_pages)
    category = classify_failure_category(
        gold_label=claim_data["label"],
        predicted_label=result.verdict,
        gold_pages=gold_pages,
        gold_page_present_in_chroma=gold_present,
        retrieval_rows=retrieval_rows,
        candidate_rows=candidate_rows,
        stance_rows=stance_rows,
    )

    return {
        "claim_id": claim_data["id"],
        "claim": claim_data["claim"],
        "gold_label": claim_data["label"],
        "predicted_label": result.verdict,
        "confidence": result.confidence,
        "failure_category": category,
        "gold_pages": serialize_gold_pages(gold_pages),
        "gold_page_present_in_current_chroma": gold_present,
        "cited_passage_ids": result.cited_passage_ids,
        "hallucinated_citations": result.hallucinated_citations,
        "atomic_verdicts": [av.to_dict() for av in result.atomic_verdicts],
        "retrieved": retrieval_rows,
        "candidates_before_rerank": candidate_rows,
        "stances": stance_rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Full-pipeline FEVER evaluation")
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
        help="Final passages to pass to the stance classifier per atomic claim.",
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
    parser.add_argument(
        "--exclude-train-overlap",
        action="store_true",
        help="Exclude FEVER dev claims whose normalized text appears in train triples.",
    )
    parser.add_argument(
        "--train-triples",
        type=str,
        default="data/processed/train.jsonl",
        help="JSONL training triples used for exact-overlap exclusion.",
    )
    parser.add_argument(
        "--verifier-train-file",
        type=str,
        default=None,
        help=(
            "Optional verifier-train JSONL (claim/label/passages) to also "
            "exclude from the dev eval. Recommended when --use-trained-verifier "
            "is set so the verifier's training claims are not silently leaked."
        ),
    )
    parser.add_argument(
        "--trace-errors-output",
        type=str,
        default=None,
        help="Optional path to write wrong-prediction traces.",
    )
    parser.add_argument(
        "--trace-error-limit",
        type=int,
        default=100,
        help="Maximum wrong-prediction traces to save (0 = all).",
    )
    parser.add_argument(
        "--skip-decomposition",
        action="store_true",
        help="Treat each FEVER claim as one atomic claim for faster eval.",
    )
    parser.add_argument(
        "--enable-title-retrieval",
        action="store_true",
        help="Add full-FEVER title/page candidates before stance classification.",
    )
    parser.add_argument(
        "--enable-reranker",
        action="store_true",
        help="Retrieve candidate_k dense/title candidates and cross-encoder rerank.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=50,
        help="Dense candidate depth before reranking when reranker/title retrieval is enabled.",
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
    parser.add_argument(
        "--stance-confidence-threshold",
        type=float,
        default=None,
        help="Eval-only stance confidence threshold override.",
    )
    parser.add_argument(
        "--include-source-title-in-stance",
        action="store_true",
        help="Prepend page/source title to each NLI premise.",
    )
    parser.add_argument(
        "--combine-same-source-evidence",
        action="store_true",
        help="Classify compact same-source evidence windows instead of isolated passages.",
    )
    parser.add_argument(
        "--same-source-max-passages",
        type=int,
        default=3,
        help="Max same-source passages to concatenate for stance classification.",
    )
    parser.add_argument(
        "--nei-confidence-floor",
        type=float,
        default=None,
        help="Eval-only synthesizer NEI floor override.",
    )
    parser.add_argument(
        "--decisive-dominance-floor",
        type=float,
        default=None,
        help="Eval-only synthesizer decisive dominance override.",
    )
    parser.add_argument(
        "--disable-refute-overrides",
        action="store_true",
        help="Disable claim-level single-refutation override.",
    )
    parser.add_argument(
        "--min-refuting-passages",
        type=int,
        default=1,
        help="Require this many refuting passages unless the single-refute floor is met.",
    )
    parser.add_argument(
        "--single-refute-confidence-floor",
        type=float,
        default=0.0,
        help="Allow one refuting passage only when its stance confidence reaches this floor.",
    )
    parser.add_argument(
        "--disable-source-relevance",
        action="store_true",
        help="Disable source-title relevance weighting in credibility scoring.",
    )
    parser.add_argument(
        "--refute-source-relevance-floor",
        type=float,
        default=None,
        help="Require refutations to come from sources at least this relevant.",
    )
    parser.add_argument(
        "--refute-conflict-margin",
        type=float,
        default=None,
        help="Source relevance margin where stronger support can block weak refutes.",
    )
    parser.add_argument(
        "--use-trained-verifier",
        action="store_true",
        help="Override the final verdict with a trained claim-level verifier.",
    )
    parser.add_argument(
        "--verifier-model-path",
        type=str,
        default="models/fever_verifier_deberta_base",
        help="Local path to the trained FEVER verifier model.",
    )
    parser.add_argument(
        "--verifier-max-length",
        type=int,
        default=384,
        help="Tokenizer max length for trained verifier inference.",
    )
    parser.add_argument(
        "--verifier-max-passages",
        type=int,
        default=None,
        help="Max retrieved passages included in trained verifier input.",
    )
    parser.add_argument(
        "--verifier-device",
        type=str,
        default=None,
        help="Optional verifier device override: cuda, cpu, or mps.",
    )
    parser.add_argument(
        "--verifier-temperature",
        type=float,
        default=None,
        help=(
            "Override the verifier softmax temperature. If unset, the verifier "
            "loads temperature.json from the model directory if present, else 1.0."
        ),
    )
    parser.add_argument(
        "--raw-predictions-output",
        type=str,
        default=None,
        help=(
            "Optional JSONL path. Per-claim raw records: gold label, baseline "
            "synthesis verdict/confidence, calibrated verifier probabilities. "
            "Use with tune_pipeline_thresholds.py to do held-out threshold tuning."
        ),
    )
    parser.add_argument(
        "--verifier-min-confidence",
        type=float,
        default=None,
        help="Downgrade trained-verifier predictions below this confidence to NEI.",
    )
    parser.add_argument(
        "--verifier-min-margin",
        type=float,
        default=None,
        help="Downgrade trained-verifier predictions below this top-2 probability margin to NEI.",
    )
    parser.add_argument(
        "--verifier-supported-threshold",
        type=float,
        default=None,
        help="Minimum confidence required to keep a trained-verifier SUPPORTED prediction.",
    )
    parser.add_argument(
        "--verifier-refuted-threshold",
        type=float,
        default=None,
        help="Minimum confidence required to keep a trained-verifier REFUTED prediction.",
    )
    parser.add_argument(
        "--verifier-nei-threshold",
        type=float,
        default=None,
        help="Minimum confidence required to keep a trained-verifier NEI prediction.",
    )
    parser.add_argument(
        "--ensemble-baseline-refute-fallback",
        action="store_true",
        help="Recover strong baseline REFUTED predictions when the verifier abstains.",
    )
    parser.add_argument(
        "--ensemble-baseline-refute-threshold",
        type=float,
        default=0.75,
        help="Min baseline confidence for --ensemble-baseline-refute-fallback.",
    )
    parser.add_argument(
        "--ensemble-verifier-refute-prob-threshold",
        type=float,
        default=0.25,
        help="Min verifier REFUTED probability for baseline refute fallback.",
    )
    args = parser.parse_args()

    exclude_claim_texts = None
    if args.exclude_train_overlap:
        sources = [args.train_triples]
        if args.verifier_train_file:
            sources.append(args.verifier_train_file)
        exclude_claim_texts = load_train_claim_texts(sources)
        logger.info(
            f"Loaded {len(exclude_claim_texts)} training claims for "
            f"decontamination from {len(sources)} source(s)"
        )

    claims, excluded_count = load_fever_dev_claims(
        args.max_claims,
        exclude_claim_texts=exclude_claim_texts,
    )

    logger.info("Initializing FactCheckAgent (this loads models lazily)...")
    agent = build_agent(args)

    verifier = None
    if args.use_trained_verifier:
        from src.claim_processing.verdict_verifier import (
            FeverVerdictVerifier,
            apply_baseline_refute_fallback,
            calibrate_verifier_prediction,
            flatten_trace_retrievals,
            override_synthesis_with_prediction,
        )

        logger.info(f"Loading trained verifier from {args.verifier_model_path}")
        verifier = FeverVerdictVerifier(
            args.verifier_model_path,
            device=args.verifier_device,
            max_length=args.verifier_max_length,
            temperature=args.verifier_temperature,
        )
        logger.info(
            "Verifier temperature in use: T = {:.4f}", verifier.temperature
        )

    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float] = []
    correct_flags: list[bool] = []
    hallucination_flags: list[bool] = []
    citation_missing_flags: list[bool] = []
    error_traces: list[dict] = []
    raw_records: list[dict] = []

    chroma_index_dir = settings.get_absolute_path(settings.chroma_persist_dir)

    logger.info(f"Running pipeline on {len(claims)} claims...")
    with ChromaPagePresence.from_index_dir(chroma_index_dir) as page_presence:
        page_presence_stats = gold_page_presence_stats(claims, page_presence)

        for i, claim_data in enumerate(tqdm(claims, desc="Pipeline eval")):
            try:
                trace = agent.check_with_trace(claim_data["claim"])
                result = trace.synthesis
                if verifier is not None:
                    baseline_result = result
                    retrievals = flatten_trace_retrievals(trace)
                    prediction = verifier.predict(
                        claim_data["claim"],
                        retrievals,
                        max_passages=args.verifier_max_passages,
                    )
                    if args.raw_predictions_output:
                        raw_records.append({
                            "claim_id": claim_data["id"],
                            "gold_label": claim_data["label"],
                            "baseline_verdict": baseline_result.verdict,
                            "baseline_confidence": float(baseline_result.confidence),
                            "verifier_probabilities": dict(prediction.probabilities),
                            "verifier_label": prediction.label,
                            "retrieved_passage_ids": [
                                r.passage.id for r in retrievals
                            ],
                        })
                    prediction = calibrate_verifier_prediction(
                        prediction,
                        min_confidence=args.verifier_min_confidence,
                        min_margin=args.verifier_min_margin,
                        supported_threshold=args.verifier_supported_threshold,
                        refuted_threshold=args.verifier_refuted_threshold,
                        nei_threshold=args.verifier_nei_threshold,
                    )
                    if args.ensemble_baseline_refute_fallback:
                        prediction = apply_baseline_refute_fallback(
                            prediction,
                            baseline_result,
                            min_baseline_confidence=(
                                args.ensemble_baseline_refute_threshold
                            ),
                            min_verifier_refute_probability=(
                                args.ensemble_verifier_refute_prob_threshold
                            ),
                        )
                    result = override_synthesis_with_prediction(
                        baseline_result,
                        prediction,
                        retrievals,
                    )
                    trace.synthesis = result
            except Exception as exc:
                logger.warning(
                    f"Claim {claim_data['id']} failed: {exc}. Defaulting to NEI."
                )
                y_true.append(claim_data["label"])
                y_pred.append("NOT_ENOUGH_INFO")
                confidences.append(0.0)
                correct_flags.append(claim_data["label"] == "NOT_ENOUGH_INFO")
                hallucination_flags.append(False)
                citation_missing_flags.append(False)
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

            should_trace = (
                args.trace_errors_output
                and pred != gold
                and (
                    args.trace_error_limit <= 0
                    or len(error_traces) < args.trace_error_limit
                )
            )
            if should_trace:
                error_traces.append(
                    make_error_trace(
                        claim_data,
                        trace,
                        agent.retriever,
                        page_presence,
                    )
                )

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

    sep = "=" * 64
    print(f"\n{sep}")
    print("  Full Pipeline Evaluation - Results")
    print(sep)
    print(f"\n  Claims evaluated : {n:,}")
    print(f"  Top-k per atomic : {args.top_k}")
    print(f"  Candidate-k      : {args.candidate_k}")
    print(f"  Adaptive mode    : {args.adaptive}")
    print(f"  Title retrieval  : {args.enable_title_retrieval}")
    print(f"  Reranker         : {args.enable_reranker}")
    print(f"  Trained verifier : {args.use_trained_verifier}")
    if args.exclude_train_overlap:
        print(f"  Train overlaps excluded : {excluded_count:,}")

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

    print(f"\n-- Retrieval Diagnostics --------------------------------------")
    if page_presence_stats["claims_with_gold_pages"]:
        print(
            "  Gold page present in current Chroma index: "
            f"{page_presence_stats['claims_with_gold_page_present']:,}/"
            f"{page_presence_stats['claims_with_gold_pages']:,} "
            f"({page_presence_stats['gold_page_present_rate']:.3f})"
        )

    print(f"\n-- Citation & Hallucination -----------------------------------")
    print(f"  Hallucinated citations : {halluc_count:,}/{n:,}  ({halluc_rate:.1%})")
    print(f"  Missing citations      : {cite_missing:,}/{n:,}  ({cite_missing_rate:.1%})")
    target_met = "YES" if halluc_rate < 0.05 else "NO"
    print(f"  Hallucination < 5%     : {target_met}")

    print(f"\n{sep}\n")

    results_dict = {
        "n_claims": n,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "index_type": "current_chroma_plus_fever_title"
        if args.enable_title_retrieval
        else "current_chroma",
        "embedding_model": settings.embedding_model,
        "adaptive": args.adaptive,
        "exclude_train_overlap": args.exclude_train_overlap,
        "excluded_train_overlap_count": excluded_count,
        "train_triples": args.train_triples if args.exclude_train_overlap else None,
        "verifier_train_file": (
            args.verifier_train_file if args.exclude_train_overlap else None
        ),
        "skip_decomposition": args.skip_decomposition,
        "enable_title_retrieval": args.enable_title_retrieval,
        "enable_reranker": args.enable_reranker,
        "reranker_model": args.reranker_model if args.enable_reranker else None,
        "title_index_path": args.title_index_path
        if args.enable_title_retrieval
        else None,
        "title_candidate_pages": args.title_candidate_pages,
        "title_candidate_k": args.title_candidate_k,
        "stance_confidence_threshold": args.stance_confidence_threshold,
        "include_source_title_in_stance": args.include_source_title_in_stance,
        "combine_same_source_evidence": args.combine_same_source_evidence,
        "same_source_max_passages": args.same_source_max_passages,
        "use_trained_verifier": args.use_trained_verifier,
        "verifier_model_path": args.verifier_model_path
        if args.use_trained_verifier
        else None,
        "verifier_max_length": args.verifier_max_length
        if args.use_trained_verifier
        else None,
        "verifier_max_passages": args.verifier_max_passages
        if args.use_trained_verifier
        else None,
        "verifier_min_confidence": args.verifier_min_confidence
        if args.use_trained_verifier
        else None,
        "verifier_min_margin": args.verifier_min_margin
        if args.use_trained_verifier
        else None,
        "verifier_supported_threshold": args.verifier_supported_threshold
        if args.use_trained_verifier
        else None,
        "verifier_refuted_threshold": args.verifier_refuted_threshold
        if args.use_trained_verifier
        else None,
        "verifier_nei_threshold": args.verifier_nei_threshold
        if args.use_trained_verifier
        else None,
        "ensemble_baseline_refute_fallback": args.ensemble_baseline_refute_fallback
        if args.use_trained_verifier
        else False,
        "ensemble_baseline_refute_threshold": (
            args.ensemble_baseline_refute_threshold
            if args.use_trained_verifier and args.ensemble_baseline_refute_fallback
            else None
        ),
        "ensemble_verifier_refute_prob_threshold": (
            args.ensemble_verifier_refute_prob_threshold
            if args.use_trained_verifier and args.ensemble_baseline_refute_fallback
            else None
        ),
        "nei_confidence_floor": args.nei_confidence_floor,
        "decisive_dominance_floor": args.decisive_dominance_floor,
        "refute_overrides": not args.disable_refute_overrides,
        "min_refuting_passages": args.min_refuting_passages,
        "single_refute_confidence_floor": args.single_refute_confidence_floor,
        "source_relevance_enabled": not args.disable_source_relevance,
        "refute_source_relevance_floor": args.refute_source_relevance_floor
        if args.refute_source_relevance_floor is not None
        else 0.55,
        "refute_conflict_margin": args.refute_conflict_margin
        if args.refute_conflict_margin is not None
        else 0.15,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "ece": ece,
        "hallucination_rate": halluc_rate,
        "citation_missing_rate": cite_missing_rate,
        **page_presence_stats,
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

    out_path = Path(args.output or "data/processed/pipeline_eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results_dict, indent=2), encoding="utf-8")
    logger.info(f"Results saved to {out_path}")

    if args.trace_errors_output:
        trace_path = Path(args.trace_errors_output)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(error_traces, indent=2), encoding="utf-8")
        logger.info(f"Wrong-prediction traces saved to {trace_path}")

    if args.raw_predictions_output and raw_records:
        raw_path = Path(args.raw_predictions_output)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w", encoding="utf-8") as f:
            for record in raw_records:
                f.write(json.dumps(record) + "\n")
        logger.info(f"Raw predictions saved to {raw_path} ({len(raw_records)} records)")


if __name__ == "__main__":
    main()
