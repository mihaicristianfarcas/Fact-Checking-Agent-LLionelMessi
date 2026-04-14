"""
Source Credibility Scorer.

Rates the reliability of each retrieved evidence passage using a weighted
combination of signals already available in the pipeline:

    1. Dataset prior    — FEVER Wikipedia passages get higher base credibility
                          than PolitiFact claim texts (encyclopedic vs editorial).
    2. Retrieval rank   — higher-ranked passages are more topically relevant.
    3. Retrieval score  — raw cosine similarity from the vector search.
    4. Stance confidence — passages where the NLI model is more confident
                          are treated as more informative (regardless of label).

The scorer does NOT fetch external metadata, call APIs, or require a model.
It is a deterministic, fast heuristic that can be swapped for a learned scorer
later without changing the downstream interface.

Downstream contract:
    Returns a list[ScoredPassage] that pairs each PassageStance with a
    credibility weight in [0, 1].  The VerdictSynthesizer consumes these
    weights when aggregating atomic-claim signals into the final verdict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.claim_processing.stance_classifier import PassageStance, StanceResult

logger = logging.getLogger(__name__)


_DATASET_PRIORS: dict[str, float] = {
    "fever": 0.90,
    "politifact": 0.70,
}
_DEFAULT_PRIOR = 0.50


@dataclass
class ScoredPassage:
    """A passage stance augmented with a credibility weight."""

    stance: PassageStance
    credibility: float

    @property
    def weighted_confidence(self) -> float:
        """Stance confidence scaled by credibility — used by the synthesizer."""
        return self.stance.confidence * self.credibility


class CredibilityScorer:
    """Heuristic source-credibility scorer.

    Args:
        dataset_priors : Override the default dataset → base credibility map.
        rank_decay     : How quickly credibility drops with rank.
                         credibility_rank_factor = 1 / (1 + rank_decay * (rank - 1))
        retrieval_weight : Weight of the retrieval-score signal (0-1).
        stance_weight    : Weight of the stance-confidence signal (0-1).
        prior_weight     : Weight of the dataset-prior signal (0-1).
    """

    def __init__(
        self,
        dataset_priors: Optional[dict[str, float]] = None,
        rank_decay: float = 0.10,
        retrieval_weight: float = 0.30,
        stance_weight: float = 0.30,
        prior_weight: float = 0.20,
        rank_weight: float = 0.20,
    ) -> None:
        self.dataset_priors = dataset_priors or _DATASET_PRIORS
        self.rank_decay = rank_decay

        total = retrieval_weight + stance_weight + prior_weight + rank_weight
        if total <= 0:
            raise ValueError("Sum of credibility weights must be positive.")
        self.w_retrieval = retrieval_weight / total
        self.w_stance = stance_weight / total
        self.w_prior = prior_weight / total
        self.w_rank = rank_weight / total

    def score(self, stance_result: StanceResult) -> list[ScoredPassage]:
        """Score every passage in a StanceResult.

        Args:
            stance_result: Output of StanceClassifier.classify() for one
                           atomic claim.

        Returns:
            One ScoredPassage per passage stance, in the same order.
        """
        scored: list[ScoredPassage] = []

        for ps in stance_result.passage_stances:
            prior = self.dataset_priors.get(ps.passage_dataset, _DEFAULT_PRIOR)
            rank_factor = 1.0 / (1.0 + self.rank_decay * (ps.retrieval_rank - 1))
            retrieval_signal = max(0.0, min(1.0, ps.retrieval_score))
            stance_signal = ps.confidence

            credibility = (
                self.w_prior * prior
                + self.w_rank * rank_factor
                + self.w_retrieval * retrieval_signal
                + self.w_stance * stance_signal
            )
            credibility = max(0.0, min(1.0, credibility))

            scored.append(ScoredPassage(stance=ps, credibility=credibility))

        logger.debug(
            "Scored %d passages for claim %r (creds: %s)",
            len(scored),
            stance_result.claim_text[:60],
            [round(s.credibility, 3) for s in scored],
        )
        return scored

    def score_batch(
        self, stance_results: list[StanceResult]
    ) -> list[list[ScoredPassage]]:
        """Score multiple StanceResults."""
        return [self.score(sr) for sr in stance_results]
