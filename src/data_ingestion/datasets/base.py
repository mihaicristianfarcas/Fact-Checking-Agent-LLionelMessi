"""Base data models and abstract interfaces for datasets."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator


class Verdict(str, Enum):
    """Standardized verdict labels across all datasets."""

    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


@dataclass
class EvidencePassage:
    """A single piece of evidence from the corpus.

    Attributes:
        id: Unique identifier for this passage
        text: The evidence text content
        source: Source identifier (Wikipedia article title, URL, etc.)
        dataset: Which dataset this came from ("fever" or "politifact")
        metadata: Additional fields (date, author, sentence_id, etc.)
    """

    id: str
    text: str
    source: str
    dataset: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "dataset": self.dataset,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvidencePassage":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            text=data["text"],
            source=data["source"],
            dataset=data["dataset"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class Claim:
    """A claim to be fact-checked.

    Attributes:
        id: Unique identifier for this claim
        text: The claim text
        verdict: Ground truth verdict (if available)
        evidence: List of evidence passages supporting the verdict
        dataset: Which dataset this came from
        metadata: Additional fields (date, speaker, context, etc.)
    """

    id: str
    text: str
    verdict: Verdict | None = None
    evidence: list[EvidencePassage] = field(default_factory=list)
    dataset: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "verdict": self.verdict.value if self.verdict else None,
            "evidence": [e.to_dict() for e in self.evidence],
            "dataset": self.dataset,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Claim":
        """Create from dictionary."""
        verdict = Verdict(data["verdict"]) if data.get("verdict") else None
        evidence = [EvidencePassage.from_dict(e) for e in data.get("evidence", [])]
        return cls(
            id=data["id"],
            text=data["text"],
            verdict=verdict,
            evidence=evidence,
            dataset=data.get("dataset", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ClaimEvidenceTriple:
    """A training/evaluation triple: claim + evidence + verdict.

    This is the format used for fine-tuning and evaluation.
    """

    claim_id: str
    claim_text: str
    evidence_passages: list[EvidencePassage]
    verdict: Verdict
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "evidence_passages": [e.to_dict() for e in self.evidence_passages],
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClaimEvidenceTriple":
        """Create from dictionary."""
        evidence = [
            EvidencePassage.from_dict(e) for e in data.get("evidence_passages", [])
        ]
        return cls(
            claim_id=data["claim_id"],
            claim_text=data["claim_text"],
            evidence_passages=evidence,
            verdict=Verdict(data["verdict"]),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )


class BaseDataset(ABC):
    """Abstract base class for dataset loaders."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name identifier."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load the dataset into memory or prepare for iteration."""
        ...

    @abstractmethod
    def iter_claims(self) -> Iterator[Claim]:
        """Iterate over all claims in the dataset."""
        ...

    @abstractmethod
    def iter_evidence(self) -> Iterator[EvidencePassage]:
        """Iterate over all evidence passages in the dataset."""
        ...

    @abstractmethod
    def get_statistics(self) -> dict:
        """Return dataset statistics (counts, label distribution, etc.)."""
        ...
