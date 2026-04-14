"""
Responsibility:
    Given one atomic claim and a list of evidence passages retrieved from the
    ChromaDB index (Task A), label each passage as SUPPORTING, REFUTING, or
    NEUTRAL with respect to the claim, and return calibrated confidence scores.

Model Choice — cross-encoder/nli-deberta-v3-base:
    We use a HuggingFace cross-encoder (discriminative NLI) rather than a
    generative LLM for this component because:

    1. Calibrated probabilities — the softmax over [entailment, neutral,
       contradiction] gives meaningful 0-1 confidence values. A generative
       model's output logits are not comparably calibrated.
    2. Zero hallucination risk — it is a classifier, not a generator.
    3. Speed — a single 512-token cross-encoder forward pass is ~10-50 ms on
       CPU; a generative API call is ~500-2000 ms.
    4. NLI is exactly the right task — the model was fine-tuned to judge
       whether a hypothesis (claim) is entailed, neutral, or contradicted by
       a premise (passage), which is precisely our use-case.
    5. Deterministic — same input always yields the same output.

    DeBERTa-v3 consistently tops NLI benchmarks (MNLI, SNLI, FEVER-NLI),
    making it the strongest off-the-shelf choice.

    NLI label → stance label mapping:
        entailment    → SUPPORTING
        neutral       → NEUTRAL
        contradiction → REFUTING

Downstream contract:
    Returns a StanceResult containing one PassageStance per RetrievalResult,
    plus aggregate verdict signals (best_label, aggregate_score) that the
    synthesis layer (Task C) can consume directly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.claim_processing.text_cleaner import clean_passages_in_retrieval_results
from src.config.settings import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Enums & Constants
# ─────────────────────────────────────────────────────────────────────────────


class StanceLabel(str, Enum):
    """
    Stance of an evidence passage relative to the claim being checked.

    Intentionally uppercase to match Task A's ground-truth verdict vocabulary
    so the synthesis layer can compare them directly.
    """

    SUPPORTING = "SUPPORTING"
    REFUTING = "REFUTING"
    NEUTRAL = "NEUTRAL"


# Threshold below which we downgrade a predicted label to NEUTRAL.
# Configurable via FACTCHECK_STANCE_CONFIDENCE_THRESHOLD env var or .env file.
_CONFIDENCE_THRESHOLD = settings.stance_confidence_threshold

# Default HuggingFace model for NLI
_DEFAULT_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"


# ─────────────────────────────────────────────────────────────────────────────
# Task A RetrievalResult Interface
# ─────────────────────────────────────────────────────────────────────────────
# We define a minimal Protocol here so this module can be developed and tested
# independently of Task A's concrete implementation.  The real EvidenceRetriever
# returns objects that satisfy these attribute contracts.


class _PassageProtocol:
    """Structural type stub for Task A's Passage object."""
    text: str
    source: str
    dataset: str
    id: str


class _RetrievalResultProtocol:
    """
    Structural type stub for Task A's RetrievalResult object.

    Task A's actual fields (confirmed):
        r.score        (float) : cosine similarity 0-1
        r.rank         (int)   : 1-indexed position in result list
        r.passage.text (str)   : evidence sentence
        r.passage.source (str) : Wikipedia title or "politifact"
        r.passage.dataset (str): "fever" or "politifact"
        r.passage.id   (str)   : unique passage ID
    """
    score: float
    rank: int
    passage: _PassageProtocol


# ─────────────────────────────────────────────────────────────────────────────
# Data Contracts
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PassageStance:
    """
    Stance classification result for a single evidence passage.

    Attributes:
        passage_id       : Task A's unique passage ID (r.passage.id).
        passage_text     : The evidence text that was classified.
        passage_source   : Wikipedia title or "politifact" (r.passage.source).
        passage_dataset  : "fever" or "politifact" (r.passage.dataset).
        retrieval_score  : Cosine similarity from Task A retriever (r.score).
        retrieval_rank   : Rank from Task A retriever (r.rank).
        stance           : SUPPORTING | REFUTING | NEUTRAL.
        confidence       : Probability of the predicted stance label (0-1).
        raw_scores       : Full {label: prob} dict for all three labels.
    """

    passage_id: str
    passage_text: str
    passage_source: str
    passage_dataset: str
    retrieval_score: float
    retrieval_rank: int
    stance: StanceLabel
    confidence: float
    raw_scores: dict[str, float] = field(default_factory=dict)

    def is_decisive(self, threshold: float = _CONFIDENCE_THRESHOLD) -> bool:
        """True if the stance is non-neutral AND confidence exceeds threshold."""
        return self.stance != StanceLabel.NEUTRAL and self.confidence >= threshold

    def __repr__(self) -> str:
        return (
            f"PassageStance(rank={self.retrieval_rank}, "
            f"stance={self.stance.value}, conf={self.confidence:.3f}, "
            f"passage={self.passage_text[:60]!r}...)"
        )


@dataclass
class StanceResult:
    """
    Full output of classifying one atomic claim against its retrieved passages.

    Attributes:
        claim_text         : The atomic claim that was classified.
        passage_stances    : One PassageStance per input RetrievalResult, in
                             the same order as the input list.
        aggregate_label    : Majority vote across decisive passages, or NEUTRAL.
        aggregate_score    : Mean confidence of decisive passages for the
                             aggregate label; 0.0 if no decisive passages.
        supporting_count   : How many passages are SUPPORTING.
        refuting_count     : How many passages are REFUTING.
        neutral_count      : How many passages are NEUTRAL.
        latency_ms         : Total inference wall-clock time in ms.
        model_name         : HuggingFace model that produced this result.
    """

    claim_text: str
    passage_stances: list[PassageStance]
    aggregate_label: StanceLabel
    aggregate_score: float
    supporting_count: int
    refuting_count: int
    neutral_count: int
    latency_ms: float
    model_name: str

    @property
    def top_supporting(self) -> list[PassageStance]:
        """Supporting passages sorted by confidence descending."""
        return sorted(
            [p for p in self.passage_stances if p.stance == StanceLabel.SUPPORTING],
            key=lambda p: p.confidence,
            reverse=True,
        )

    @property
    def top_refuting(self) -> list[PassageStance]:
        """Refuting passages sorted by confidence descending."""
        return sorted(
            [p for p in self.passage_stances if p.stance == StanceLabel.REFUTING],
            key=lambda p: p.confidence,
            reverse=True,
        )

    def to_dict(self) -> dict:
        """Serialisable summary suitable for logging or downstream JSON."""
        return {
            "claim_text": self.claim_text,
            "aggregate_label": self.aggregate_label.value,
            "aggregate_score": round(self.aggregate_score, 4),
            "supporting_count": self.supporting_count,
            "refuting_count": self.refuting_count,
            "neutral_count": self.neutral_count,
            "latency_ms": round(self.latency_ms, 1),
            "passages": [
                {
                    "rank": ps.retrieval_rank,
                    "passage_id": ps.passage_id,
                    "stance": ps.stance.value,
                    "confidence": round(ps.confidence, 4),
                    "retrieval_score": round(ps.retrieval_score, 4),
                    "source": ps.passage_source,
                }
                for ps in self.passage_stances
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main Class
# ─────────────────────────────────────────────────────────────────────────────


class StanceClassifier:
    """
    Cross-encoder NLI stance classifier.

    Classifies each retrieved evidence passage as SUPPORTING, REFUTING, or
    NEUTRAL with respect to an atomic claim.

    Uses ``cross-encoder/nli-deberta-v3-base`` by default — the highest-scoring
    publicly available NLI cross-encoder on the FEVER-NLI benchmark.

    Usage::

        # Initialise once; the model is cached after the first load.
        classifier = StanceClassifier()

        # ── Core loop: iterate over decomposed atomic claims ──────────────
        for atomic_claim in decomposition_result.atomic_claims:

            # Task A handoff → retrieve passages
            retrievals: list[RetrievalResult] = retriever.retrieve(
                atomic_claim.text, top_k=5
            )

            # Task B → classify stance of each passage
            result: StanceResult = classifier.classify(
                claim=atomic_claim.text,
                retrievals=retrievals,
            )

            print(result.aggregate_label)   # e.g. StanceLabel.SUPPORTING
            print(result.top_supporting)    # ranked list of supporting passages

    Args:
        model_name          : HuggingFace model ID or local path.
        device              : "cuda", "mps", or "cpu" (auto-detected if None).
        batch_size          : Number of (claim, passage) pairs per forward pass.
        confidence_threshold: Min probability to treat a label as decisive;
                              below this, the label is collapsed to NEUTRAL.
        max_length          : Max tokenizer sequence length (model cap is 512).
    """

    # DeBERTa-v3 NLI label order as returned by the HuggingFace checkpoint.
    # IMPORTANT: verify with `tokenizer.config.id2label` if you swap models.
    _HF_LABEL_ORDER = ["contradiction", "entailment", "neutral"]

    # Mapping from HuggingFace NLI labels → StanceLabel
    _NLI_TO_STANCE: dict[str, StanceLabel] = {
        "entailment": StanceLabel.SUPPORTING,
        "neutral": StanceLabel.NEUTRAL,
        "contradiction": StanceLabel.REFUTING,
    }

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        batch_size: int = 16,
        confidence_threshold: float = _CONFIDENCE_THRESHOLD,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.max_length = max_length

        # Device selection
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        logger.info(
            "StanceClassifier: loading %s on %s …", self.model_name, self.device
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.device)
        self._model.eval()

        # Determine label order from the model's own config to be safe.
        id2label: dict[int, str] = self._model.config.id2label  # type: ignore[attr-defined]
        self._label_order: list[str] = [id2label[i] for i in sorted(id2label)]
        logger.info(
            "StanceClassifier ready. Label order from model config: %s",
            self._label_order,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(
        self,
        claim: str,
        retrievals: list,  # list[RetrievalResult] from Task A
    ) -> StanceResult:
        """
        Classify the stance of each retrieved passage relative to *claim*.

        This is the primary integration point with Task A.  It accepts the
        list of ``RetrievalResult`` objects returned by
        ``EvidenceRetriever.retrieve()`` directly.

        Args:
            claim      : An atomic claim string (from ClaimDecomposer).
            retrievals : list[RetrievalResult] from Task A's EvidenceRetriever.
                         May be empty — returns a NEUTRAL result with no passages.

        Returns:
            StanceResult with per-passage stances and an aggregate verdict.
        """
        if not retrievals:
            logger.warning("classify() called with no retrievals for claim: %r", claim)
            return self._empty_result(claim)

        clean_passages_in_retrieval_results(retrievals)

        start = time.perf_counter()

        # Build (premise, hypothesis) pairs for the NLI model.
        # Convention: premise = evidence passage, hypothesis = claim to verify.
        pairs: list[tuple[str, str]] = [
            (r.passage.text, claim) for r in retrievals
        ]

        # Run inference in batches.
        all_probs: list[dict[str, float]] = self._batch_infer(pairs)

        latency_ms = (time.perf_counter() - start) * 1000

        # Package results.
        passage_stances: list[PassageStance] = []
        for retrieval, probs in zip(retrievals, all_probs):
            ps = self._build_passage_stance(retrieval, probs)
            passage_stances.append(ps)

        aggregate_label, aggregate_score = self._aggregate(passage_stances)

        counts = {
            StanceLabel.SUPPORTING: 0,
            StanceLabel.REFUTING: 0,
            StanceLabel.NEUTRAL: 0,
        }
        for ps in passage_stances:
            counts[ps.stance] += 1

        result = StanceResult(
            claim_text=claim,
            passage_stances=passage_stances,
            aggregate_label=aggregate_label,
            aggregate_score=aggregate_score,
            supporting_count=counts[StanceLabel.SUPPORTING],
            refuting_count=counts[StanceLabel.REFUTING],
            neutral_count=counts[StanceLabel.NEUTRAL],
            latency_ms=latency_ms,
            model_name=self.model_name,
        )

        logger.info(
            "Claim %r — %d passages: %d supporting, %d refuting, %d neutral "
            "(aggregate=%s, score=%.3f, %.0f ms)",
            claim[:60],
            len(retrievals),
            result.supporting_count,
            result.refuting_count,
            result.neutral_count,
            result.aggregate_label.value,
            result.aggregate_score,
            latency_ms,
        )
        return result

    def classify_batch(
        self,
        claims_and_retrievals: list[tuple[str, list]],
    ) -> list[StanceResult]:
        """
        Classify multiple (claim, retrievals) pairs.

        Args:
            claims_and_retrievals : List of (claim_text, list[RetrievalResult]).

        Returns:
            One StanceResult per input pair, in the same order.
        """
        return [self.classify(claim, retrievals) for claim, retrievals in claims_and_retrievals]

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _batch_infer(
        self, pairs: list[tuple[str, str]]
    ) -> list[dict[str, float]]:
        """
        Run the NLI model over a list of (premise, hypothesis) pairs in
        mini-batches.  Returns one probability dict per pair.

        The cross-encoder tokenizer expects the two sequences to be passed as
        separate arguments so it inserts the correct [SEP] token.
        """
        all_probs: list[dict[str, float]] = []

        for batch_start in range(0, len(pairs), self.batch_size):
            batch = pairs[batch_start: batch_start + self.batch_size]
            premises = [p for p, _ in batch]
            hypotheses = [h for _, h in batch]

            encoding = self._tokenizer(
                premises,
                hypotheses,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self._model(**encoding).logits  # (batch, num_labels)

            probs_tensor = torch.softmax(logits, dim=-1).cpu()  # (batch, num_labels)

            for row in probs_tensor:
                prob_dict = {
                    label: float(row[i])
                    for i, label in enumerate(self._label_order)
                }
                all_probs.append(prob_dict)

        return all_probs

    def _build_passage_stance(
        self,
        retrieval: "_RetrievalResultProtocol",
        probs: dict[str, float],
    ) -> PassageStance:
        """
        Convert raw NLI probabilities + Task A RetrievalResult into a
        PassageStance, applying the confidence threshold.
        """
        # Find the highest-probability NLI label.
        best_nli_label = max(probs, key=lambda k: probs[k])
        best_prob = probs[best_nli_label]

        # Map to StanceLabel; downgrade to NEUTRAL if below threshold.
        raw_stance = self._NLI_TO_STANCE[best_nli_label]
        if raw_stance != StanceLabel.NEUTRAL and best_prob < self.confidence_threshold:
            logger.debug(
                "Downgrading %s (%.3f < threshold %.3f) to NEUTRAL for passage %r",
                raw_stance.value,
                best_prob,
                self.confidence_threshold,
                retrieval.passage.id,
            )
            final_stance = StanceLabel.NEUTRAL
            final_confidence = probs.get("neutral", best_prob)
        else:
            final_stance = raw_stance
            final_confidence = best_prob

        # Build the stance-keyed score dict for downstream consumers.
        raw_scores = {
            StanceLabel.SUPPORTING.value: probs.get("entailment", 0.0),
            StanceLabel.REFUTING.value: probs.get("contradiction", 0.0),
            StanceLabel.NEUTRAL.value: probs.get("neutral", 0.0),
        }

        return PassageStance(
            # ── Fields sourced directly from Task A's RetrievalResult ──────
            passage_id=retrieval.passage.id,
            passage_text=retrieval.passage.text,
            passage_source=retrieval.passage.source,
            passage_dataset=retrieval.passage.dataset,
            retrieval_score=retrieval.score,
            retrieval_rank=retrieval.rank,
            # ── Fields produced by this classifier ─────────────────────────
            stance=final_stance,
            confidence=final_confidence,
            raw_scores=raw_scores,
        )

    def _aggregate(
        self, passage_stances: list[PassageStance]
    ) -> tuple[StanceLabel, float]:
        """
        Compute an aggregate verdict from all passage stances.

        Strategy:
        1. Collect only "decisive" passages (SUPPORTING or REFUTING above threshold).
        2. If there are decisive passages, take the label with the higher total
           weighted score (sum of confidence × retrieval_score).
        3. If no decisive passages exist, return NEUTRAL with score 0.0.

        The retrieval_score weight means passages that are more semantically
        similar to the claim carry more weight in the aggregate.
        """
        decisive = [p for p in passage_stances if p.is_decisive(self.confidence_threshold)]

        if not decisive:
            return StanceLabel.NEUTRAL, 0.0

        # Weighted score per label.
        label_scores: dict[StanceLabel, float] = {
            StanceLabel.SUPPORTING: 0.0,
            StanceLabel.REFUTING: 0.0,
        }
        label_counts: dict[StanceLabel, int] = {
            StanceLabel.SUPPORTING: 0,
            StanceLabel.REFUTING: 0,
        }

        for ps in decisive:
            if ps.stance in label_scores:
                weight = ps.confidence * ps.retrieval_score
                label_scores[ps.stance] += weight
                label_counts[ps.stance] += 1

        best_label = max(label_scores, key=lambda k: label_scores[k])
        count = label_counts[best_label]
        raw_score = label_scores[best_label] / count if count else 0.0

        return best_label, raw_score

    def _empty_result(self, claim: str) -> StanceResult:
        """Return a NEUTRAL StanceResult when no passages were retrieved."""
        return StanceResult(
            claim_text=claim,
            passage_stances=[],
            aggregate_label=StanceLabel.NEUTRAL,
            aggregate_score=0.0,
            supporting_count=0,
            refuting_count=0,
            neutral_count=0,
            latency_ms=0.0,
            model_name=self.model_name,
        )
    def classify_from_retriever(self, claim: str, results: list) -> StanceResult:
        """
        Directly accepts the list of RetrievalResult objects from Task A's EvidenceRetriever.
        """
        # This matches the 'r.passage.text' structure in the README
        return self.classify(claim, results)