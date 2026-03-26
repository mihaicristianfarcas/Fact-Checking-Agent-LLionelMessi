"""Ground-truth triples module."""

from .triple_generator import (
    ClaimEvidenceTriple,
    TripleGenerator,
    generate_training_triples,
)

__all__ = ["ClaimEvidenceTriple", "TripleGenerator", "generate_training_triples"]
