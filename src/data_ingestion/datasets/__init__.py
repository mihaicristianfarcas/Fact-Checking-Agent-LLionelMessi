"""Dataset loaders."""

from .base import BaseDataset, Claim, ClaimEvidenceTriple, EvidencePassage, Verdict
from .fever import FeverDataset
from .politifact import PolitifactDataset, load_combined_politifact

__all__ = [
    "BaseDataset",
    "Claim",
    "ClaimEvidenceTriple",
    "EvidencePassage",
    "Verdict",
    "FeverDataset",
    "PolitifactDataset",
    "load_combined_politifact",
]
