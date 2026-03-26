"""PolitiFact/LIAR dataset loader.

The LIAR dataset contains 12,836 short statements from PolitiFact
with six fine-grained labels: pants-fire, false, barely-true,
half-true, mostly-true, and true.

We map these to the three-class schema: SUPPORTED, REFUTED, NOT_ENOUGH_INFO.
"""

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from loguru import logger

from .base import BaseDataset, Claim, EvidencePassage, Verdict


# Map PolitiFact 6-tier ratings to 3-tier verdict
# true, mostly-true, half-true -> SUPPORTED (leaning positive)
# barely-true, false, pants-fire -> REFUTED (leaning negative)
POLITIFACT_LABEL_MAP = {
    "true": Verdict.SUPPORTED,
    "mostly-true": Verdict.SUPPORTED,
    "half-true": Verdict.NOT_ENOUGH_INFO,  # Ambiguous - map to NEI
    "barely-true": Verdict.REFUTED,
    "false": Verdict.REFUTED,
    "pants-fire": Verdict.REFUTED,
}


class PolitifactDataset(BaseDataset):
    """Loader for PolitiFact data via the LIAR dataset.

    Example:
        ```python
        dataset = PolitifactDataset(split="train")
        dataset.load()

        for claim in dataset.iter_claims():
            print(f"{claim.verdict}: {claim.text}")
        ```
    """

    def __init__(
        self,
        split: str = "train",
        cache_dir: str | None = None,
        max_samples: int | None = None,
    ):
        """Initialize the PolitiFact dataset loader.

        Args:
            split: Dataset split ("train", "validation", "test")
            cache_dir: Directory to cache downloaded data
            max_samples: Maximum number of samples to load (None for all)
        """
        self.split = split
        self.cache_dir = cache_dir
        self.max_samples = max_samples
        self._dataset = None
        self._claims: list[Claim] = []

    @property
    def name(self) -> str:
        return "politifact"

    def load(self) -> None:
        """Load the LIAR dataset from HuggingFace."""
        logger.info(f"Loading LIAR/PolitiFact dataset (split={self.split})...")

        self._dataset = load_dataset(
            "liar",
            split=self.split,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            verification_mode="no_checks",
        )

        # Parse into Claim objects
        self._claims = []
        for idx, item in enumerate(self._dataset):
            if self.max_samples and idx >= self.max_samples:
                break

            claim = self._parse_claim(item, idx)
            if claim:
                self._claims.append(claim)

        logger.info(
            f"Loaded {len(self._claims)} claims from LIAR/PolitiFact {self.split}"
        )

    def _parse_claim(self, item: dict, idx: int) -> Claim | None:
        """Parse a LIAR dataset item into a Claim object."""
        label_str = item.get("label")
        if label_str is None:
            return None

        # Map numeric label to string
        label_names = [
            "false",
            "half-true",
            "mostly-true",
            "true",
            "barely-true",
            "pants-fire",
        ]
        if isinstance(label_str, int):
            label_str = label_names[label_str] if label_str < len(label_names) else None

        verdict = POLITIFACT_LABEL_MAP.get(label_str)
        if verdict is None:
            return None

        # Extract metadata
        speaker = item.get("speaker", "")
        job_title = item.get("job_title", "")
        state = item.get("state_info", "")
        party = item.get("party_affiliation", "")
        context = item.get("context", "")
        subject = item.get("subject", "")

        return Claim(
            id=f"politifact_{item.get('id', idx)}",
            text=item.get("statement", ""),
            verdict=verdict,
            evidence=[],  # LIAR doesn't include evidence text
            dataset="politifact",
            metadata={
                "original_label": label_str,
                "speaker": speaker,
                "job_title": job_title,
                "state": state,
                "party": party,
                "context": context,
                "subject": subject,
            },
        )

    def iter_claims(self) -> Iterator[Claim]:
        """Iterate over all claims in the dataset."""
        if not self._claims:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        yield from self._claims

    def iter_evidence(self) -> Iterator[EvidencePassage]:
        """Iterate over evidence passages.

        Note: LIAR dataset doesn't include evidence passages,
        so this returns an empty iterator.
        """
        return iter([])

    def get_statistics(self) -> dict:
        """Return dataset statistics."""
        if not self._claims:
            return {"loaded": False}

        # Count by original label and mapped verdict
        original_counts: dict[str, int] = {}
        verdict_counts: dict[str, int] = {}

        for claim in self._claims:
            orig_label = claim.metadata.get("original_label", "unknown")
            original_counts[orig_label] = original_counts.get(orig_label, 0) + 1

            if claim.verdict:
                verdict_counts[claim.verdict.value] = (
                    verdict_counts.get(claim.verdict.value, 0) + 1
                )

        return {
            "loaded": True,
            "split": self.split,
            "total_claims": len(self._claims),
            "original_label_distribution": original_counts,
            "verdict_distribution": verdict_counts,
        }


def load_combined_politifact(
    cache_dir: str | None = None,
    max_samples: int | None = None,
) -> list[Claim]:
    """Load all PolitiFact splits combined.

    Args:
        cache_dir: Directory to cache downloaded data
        max_samples: Maximum total samples to load

    Returns:
        List of all claims across train/val/test splits
    """
    all_claims = []
    samples_per_split = max_samples // 3 if max_samples else None

    for split in ["train", "validation", "test"]:
        dataset = PolitifactDataset(
            split=split,
            cache_dir=cache_dir,
            max_samples=samples_per_split,
        )
        dataset.load()
        all_claims.extend(list(dataset.iter_claims()))

        if max_samples and len(all_claims) >= max_samples:
            break

    return all_claims[:max_samples] if max_samples else all_claims
