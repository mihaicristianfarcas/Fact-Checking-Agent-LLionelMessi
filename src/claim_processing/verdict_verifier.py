"""Trained claim-level FEVER verifier.

This module provides the shared input format for training and inference:

    Claim: <claim>

    Evidence:
    [1] Source: <page title>
    <passage text>

The verifier predicts the final FEVER verdict from the claim plus the retrieved
evidence block. It never generates citations; citations remain restricted to
retrieved passage IDs selected by the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from loguru import logger

from src.data_ingestion.retriever.evidence_retriever import RetrievalResult
from src.synthesis.verdict_synthesizer import AtomicVerdict, SynthesisResult
from src.utils.text import display_fever_source

VERIFIER_LABELS = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(VERIFIER_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class VerifierPrediction:
    """One trained-verifier prediction."""

    label: str
    confidence: float
    probabilities: dict[str, float]


def build_evidence_text(
    retrievals: Iterable[RetrievalResult],
    *,
    max_passages: int | None = None,
) -> str:
    """Format retrieved evidence passages for the FEVER verifier."""
    blocks: list[str] = []
    for idx, retrieval in enumerate(retrievals, start=1):
        if max_passages is not None and idx > max_passages:
            break

        passage = retrieval.passage
        source = display_fever_source(passage.source)
        text = " ".join((passage.text or "").split())
        if not text:
            continue
        blocks.append(f"[{idx}] Source: {source}\n{text}")

    if not blocks:
        return "No retrieved evidence."
    return "\n\n".join(blocks)


def build_verifier_input(
    claim: str,
    retrievals: Iterable[RetrievalResult],
    *,
    max_passages: int | None = None,
) -> str:
    """Build the exact text consumed by the verifier model."""
    clean_claim = " ".join((claim or "").split())
    evidence_text = build_evidence_text(retrievals, max_passages=max_passages)
    return f"Claim: {clean_claim}\n\nEvidence:\n{evidence_text}"


def flatten_trace_retrievals(trace) -> list[RetrievalResult]:
    """Return trace retrievals in deterministic atomic-claim/rank order."""
    retrievals: list[RetrievalResult] = []
    seen_ids: set[str] = set()
    for atomic_claim in trace.retrievals:
        for retrieval in sorted(trace.retrievals[atomic_claim], key=lambda r: r.rank):
            pid = retrieval.passage.id
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            retrievals.append(retrieval)
    return retrievals


class FeverVerdictVerifier:
    """Load and run a fine-tuned DeBERTa FEVER verifier."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str | None = None,
        max_length: int = 384,
        temperature: float | None = None,
    ) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from src.model_training.calibration import load_temperature_sidecar
        from src.utils.device import pick_device

        if device is None:
            device = pick_device()

        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

        id2label = getattr(self.model.config, "id2label", None) or ID_TO_LABEL
        self.id2label = {
            int(idx): _normalize_label(label) for idx, label in id2label.items()
        }

        if temperature is None:
            try:
                sidecar = load_temperature_sidecar(model_path)
            except (OSError, ValueError) as exc:
                logger.warning(
                    "Failed to load temperature sidecar from {}: {}", model_path, exc
                )
                sidecar = None
            self.temperature = sidecar if sidecar is not None else 1.0
        else:
            self.temperature = float(temperature)

    def predict_text(self, input_text: str) -> VerifierPrediction:
        """Predict a verdict from an already-formatted verifier input."""
        import torch

        encoded = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits[0]
            scaled = logits / float(self.temperature)
            probs = torch.softmax(scaled, dim=-1).detach().cpu().tolist()

        probabilities = {
            self.id2label.get(idx, ID_TO_LABEL[idx]): float(prob)
            for idx, prob in enumerate(probs)
        }
        label = max(probabilities, key=probabilities.get)
        return VerifierPrediction(
            label=label,
            confidence=probabilities[label],
            probabilities=probabilities,
        )

    def predict(
        self,
        claim: str,
        retrievals: Iterable[RetrievalResult],
        *,
        max_passages: int | None = None,
    ) -> VerifierPrediction:
        """Predict a verdict from a claim and retrieved evidence passages."""
        input_text = build_verifier_input(
            claim,
            retrievals,
            max_passages=max_passages,
        )
        return self.predict_text(input_text)


def override_synthesis_with_prediction(
    original_result: SynthesisResult,
    prediction: VerifierPrediction,
    retrievals: Iterable[RetrievalResult],
    *,
    max_citations: int = 2,
) -> SynthesisResult:
    """Replace the final verdict while keeping citations retrieved-only."""
    retrieved_ids = _unique_retrieved_ids(retrievals)
    cited_ids: list[str] = []

    if prediction.label != "NOT_ENOUGH_INFO":
        valid_existing = [
            cid for cid in original_result.cited_passage_ids if cid in retrieved_ids
        ]
        cited_ids = (valid_existing or retrieved_ids)[:max_citations]

    atomic = AtomicVerdict(
        claim_text=original_result.original_claim,
        verdict=prediction.label,
        confidence=prediction.confidence,
        cited_passages=cited_ids,
        supporting_ids=cited_ids if prediction.label == "SUPPORTED" else [],
        refuting_ids=cited_ids if prediction.label == "REFUTED" else [],
    )

    return SynthesisResult(
        original_claim=original_result.original_claim,
        verdict=prediction.label,
        confidence=prediction.confidence,
        explanation=_build_verifier_explanation(prediction, cited_ids),
        cited_passage_ids=cited_ids,
        atomic_verdicts=[atomic],
        all_retrieved_ids=retrieved_ids,
    )


def calibrate_verifier_prediction(
    prediction: VerifierPrediction,
    *,
    min_confidence: float | None = None,
    min_margin: float | None = None,
    supported_threshold: float | None = None,
    refuted_threshold: float | None = None,
    nei_threshold: float | None = None,
) -> VerifierPrediction:
    """Apply conservative eval-time thresholds to a verifier prediction.

    Low-confidence or low-margin decisive labels are downgraded to
    NOT_ENOUGH_INFO. This improves abstention behavior without retraining.
    """
    thresholds = {
        "SUPPORTED": supported_threshold,
        "REFUTED": refuted_threshold,
        "NOT_ENOUGH_INFO": nei_threshold,
    }
    threshold = thresholds.get(prediction.label)

    should_abstain = False
    if min_confidence is not None and prediction.confidence < min_confidence:
        should_abstain = True
    if threshold is not None and prediction.confidence < threshold:
        should_abstain = True
    if min_margin is not None and probability_margin(prediction) < min_margin:
        should_abstain = True

    if not should_abstain or prediction.label == "NOT_ENOUGH_INFO":
        return prediction

    probabilities = dict(prediction.probabilities)
    nei_confidence = probabilities.get("NOT_ENOUGH_INFO", 0.0)
    return VerifierPrediction(
        label="NOT_ENOUGH_INFO",
        confidence=nei_confidence,
        probabilities=probabilities,
    )


def apply_baseline_refute_fallback(
    prediction: VerifierPrediction,
    baseline_result: SynthesisResult,
    *,
    min_baseline_confidence: float = 0.75,
    min_verifier_refute_probability: float = 0.25,
) -> VerifierPrediction:
    """Use the original NLI pipeline to recover strong missed refutations.

    The trained verifier improved overall accuracy but tends to under-call
    REFUTED. The old pipeline has higher REFUTED recall but lower precision, so
    this fallback only fires when both systems give some refutation signal.
    """
    if prediction.label != "NOT_ENOUGH_INFO":
        return prediction
    if baseline_result.verdict != "REFUTED":
        return prediction
    if baseline_result.confidence < min_baseline_confidence:
        return prediction

    refute_prob = prediction.probabilities.get("REFUTED", 0.0)
    if refute_prob < min_verifier_refute_probability:
        return prediction

    new_refute = max(refute_prob, baseline_result.confidence)
    new_refute = min(new_refute, 1.0)
    other_total = sum(
        p for label, p in prediction.probabilities.items() if label != "REFUTED"
    )

    probabilities: dict[str, float] = {}
    if other_total > 0.0:
        remaining_mass = max(0.0, 1.0 - new_refute)
        for label, p in prediction.probabilities.items():
            if label == "REFUTED":
                probabilities[label] = new_refute
            else:
                probabilities[label] = p * remaining_mass / other_total
    else:
        for label in prediction.probabilities:
            probabilities[label] = 1.0 if label == "REFUTED" else 0.0
    return VerifierPrediction(
        label="REFUTED",
        confidence=probabilities["REFUTED"],
        probabilities=probabilities,
    )


def probability_margin(prediction: VerifierPrediction) -> float:
    """Difference between the top two verifier probabilities."""
    probs = sorted(prediction.probabilities.values(), reverse=True)
    if len(probs) < 2:
        return prediction.confidence
    return float(probs[0] - probs[1])


def _unique_retrieved_ids(retrievals: Iterable[RetrievalResult]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for retrieval in retrievals:
        pid = retrieval.passage.id
        if pid in seen:
            continue
        seen.add(pid)
        ids.append(pid)
    return ids


def _build_verifier_explanation(
    prediction: VerifierPrediction,
    cited_ids: list[str],
) -> str:
    if prediction.label == "NOT_ENOUGH_INFO":
        return (
            "The trained verifier found that the retrieved evidence does not "
            "contain enough information to verify or refute the claim."
        )

    citations = ", ".join(f"[{pid}]" for pid in cited_ids)
    action = "supported" if prediction.label == "SUPPORTED" else "refuted"
    if citations:
        return (
            f"The trained verifier predicts the claim is {action} "
            f"(confidence: {prediction.confidence:.0%}) based on {citations}."
        )
    return (
        f"The trained verifier predicts the claim is {action} "
        f"(confidence: {prediction.confidence:.0%})."
    )




def _normalize_label(label: str) -> str:
    normalized = str(label).upper().replace(" ", "_")
    if normalized == "SUPPORTS":
        return "SUPPORTED"
    if normalized == "REFUTES":
        return "REFUTED"
    if normalized in LABEL_TO_ID:
        return normalized
    return normalized
