"""
Public API:

    from claim_processing import (
        ClaimDecomposer,
        StanceClassifier,
        AtomicClaim,
        DecompositionResult,
        PassageStance,
        StanceResult,
        StanceLabel,
    )
"""

from .decomposer import AtomicClaim, ClaimDecomposer, DecompositionResult
from .stance_classifier import PassageStance, StanceClassifier, StanceLabel, StanceResult

__all__ = [
    # Decomposer
    "ClaimDecomposer",
    "AtomicClaim",
    "DecompositionResult",
    # Stance Classifier
    "StanceClassifier",
    "PassageStance",
    "StanceResult",
    "StanceLabel",
]