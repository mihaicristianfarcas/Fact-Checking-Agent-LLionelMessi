"""Source credibility scoring for the fact-checking pipeline."""

from .credibility_scorer import CredibilityScorer, ScoredPassage

__all__ = [
    "CredibilityScorer",
    "ScoredPassage",
]
