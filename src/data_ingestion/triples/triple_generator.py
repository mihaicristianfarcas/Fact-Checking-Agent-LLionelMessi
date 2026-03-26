"""Ground-truth triple generation for training and evaluation.

Generates claim–evidence–verdict triples from FEVER and PolitiFact
datasets for use in fine-tuning and evaluation.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from tqdm import tqdm

from ..datasets.base import Claim, ClaimEvidenceTriple, EvidencePassage, Verdict
from ..datasets.fever import FeverDataset
from ..datasets.politifact import PolitifactDataset


class TripleGenerator:
    """Generate and export claim-evidence-verdict triples.

    Example:
        ```python
        generator = TripleGenerator()
        generator.load_fever(split="train")
        generator.load_politifact()

        # Export for fine-tuning
        generator.export_json("data/processed/triples.json")
        generator.export_splits("data/processed/", train_ratio=0.8)
        ```
    """

    def __init__(self):
        self._triples: list[ClaimEvidenceTriple] = []

    def load_fever(
        self,
        split: str = "train",
        load_wiki: bool = True,
        max_samples: int | None = None,
    ) -> int:
        """Load triples from FEVER dataset.

        Args:
            split: Dataset split to load
            load_wiki: Whether to load Wikipedia pages for evidence text
            max_samples: Maximum number of samples to load

        Returns:
            Number of triples loaded
        """
        logger.info(f"Loading FEVER {split} for triple generation...")

        dataset = FeverDataset(split=split)
        dataset.load()

        if load_wiki:
            dataset.load_wiki_pages()

        count = 0
        for claim in tqdm(dataset.iter_claims(), desc="Processing FEVER claims"):
            if max_samples and count >= max_samples:
                break

            if claim.verdict is None:
                continue

            # Only include claims with actual evidence text
            evidence_with_text = [e for e in claim.evidence if e.text.strip()]

            triple = ClaimEvidenceTriple(
                claim_id=claim.id,
                claim_text=claim.text,
                evidence_passages=evidence_with_text,
                verdict=claim.verdict,
                confidence=1.0,  # FEVER labels are definitive
                metadata={
                    "dataset": "fever",
                    "split": split,
                    "has_evidence": len(evidence_with_text) > 0,
                },
            )
            self._triples.append(triple)
            count += 1

        logger.info(f"Loaded {count} FEVER triples")
        return count

    def load_politifact(
        self,
        max_samples: int | None = None,
    ) -> int:
        """Load triples from PolitiFact/LIAR dataset.

        Note: LIAR doesn't include evidence passages, so these triples
        will have empty evidence lists. They're useful for verdict
        prediction training but not for evidence retrieval evaluation.

        Args:
            max_samples: Maximum number of samples to load

        Returns:
            Number of triples loaded
        """
        logger.info("Loading PolitiFact/LIAR for triple generation...")

        count = 0
        for split in ["train", "validation", "test"]:
            dataset = PolitifactDataset(split=split, max_samples=max_samples)
            dataset.load()

            for claim in dataset.iter_claims():
                if max_samples and count >= max_samples:
                    break

                if claim.verdict is None:
                    continue

                triple = ClaimEvidenceTriple(
                    claim_id=claim.id,
                    claim_text=claim.text,
                    evidence_passages=[],  # LIAR has no evidence
                    verdict=claim.verdict,
                    confidence=0.8,  # PolitiFact labels have some subjectivity
                    metadata={
                        "dataset": "politifact",
                        "split": split,
                        "speaker": claim.metadata.get("speaker", ""),
                        "original_label": claim.metadata.get("original_label", ""),
                    },
                )
                self._triples.append(triple)
                count += 1

        logger.info(f"Loaded {count} PolitiFact triples")
        return count

    def get_triples(self) -> list[ClaimEvidenceTriple]:
        """Get all loaded triples."""
        return self._triples

    def iter_triples(self) -> Iterator[ClaimEvidenceTriple]:
        """Iterate over all triples."""
        yield from self._triples

    def get_statistics(self) -> dict:
        """Get statistics about loaded triples."""
        if not self._triples:
            return {"total": 0}

        verdict_counts: dict[str, int] = {}
        dataset_counts: dict[str, int] = {}
        evidence_counts = {"with_evidence": 0, "without_evidence": 0}

        for triple in self._triples:
            # Verdict distribution
            v = triple.verdict.value
            verdict_counts[v] = verdict_counts.get(v, 0) + 1

            # Dataset distribution
            ds = triple.metadata.get("dataset", "unknown")
            dataset_counts[ds] = dataset_counts.get(ds, 0) + 1

            # Evidence availability
            if triple.evidence_passages:
                evidence_counts["with_evidence"] += 1
            else:
                evidence_counts["without_evidence"] += 1

        return {
            "total": len(self._triples),
            "verdict_distribution": verdict_counts,
            "dataset_distribution": dataset_counts,
            "evidence_counts": evidence_counts,
        }

    def export_json(self, output_path: str | Path) -> None:
        """Export triples to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [t.to_dict() for t in self._triples]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(data)} triples to {output_path}")

    def export_jsonl(self, output_path: str | Path) -> None:
        """Export triples to JSON Lines file (one JSON object per line).

        Args:
            output_path: Path to output JSONL file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for triple in self._triples:
                f.write(json.dumps(triple.to_dict()) + "\n")

        logger.info(f"Exported {len(self._triples)} triples to {output_path}")

    def export_parquet(self, output_path: str | Path) -> None:
        """Export triples to Parquet file.

        Args:
            output_path: Path to output Parquet file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Flatten for DataFrame
        records = []
        for t in self._triples:
            record = {
                "claim_id": t.claim_id,
                "claim_text": t.claim_text,
                "verdict": t.verdict.value,
                "confidence": t.confidence,
                "evidence_count": len(t.evidence_passages),
                "evidence_texts": [e.text for e in t.evidence_passages],
                "evidence_sources": [e.source for e in t.evidence_passages],
            }
            record.update(t.metadata)
            records.append(record)

        df = pd.DataFrame(records)
        df.to_parquet(output_path, index=False)

        logger.info(f"Exported {len(records)} triples to {output_path}")

    def export_splits(
        self,
        output_dir: str | Path,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        format: str = "jsonl",
    ) -> dict[str, int]:
        """Export triples with train/val/test splits.

        Args:
            output_dir: Directory for output files
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            seed: Random seed for reproducibility
            format: Output format ("jsonl" or "parquet")

        Returns:
            Dict with counts for each split
        """
        import random

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Shuffle
        triples = self._triples.copy()
        random.seed(seed)
        random.shuffle(triples)

        # Split
        n = len(triples)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        splits = {
            "train": triples[:train_end],
            "val": triples[train_end:val_end],
            "test": triples[val_end:],
        }

        counts = {}
        for split_name, split_data in splits.items():
            # Temporarily set triples for export
            original = self._triples
            self._triples = split_data

            if format == "jsonl":
                self.export_jsonl(output_dir / f"{split_name}.jsonl")
            elif format == "parquet":
                self.export_parquet(output_dir / f"{split_name}.parquet")

            self._triples = original
            counts[split_name] = len(split_data)

        logger.info(f"Exported splits: {counts}")
        return counts


def generate_training_triples(
    output_dir: str | Path = "data/processed/triples",
    fever_split: str = "train",
    max_fever: int | None = None,
    max_politifact: int | None = None,
) -> dict:
    """Convenience function to generate and export training triples.

    Args:
        output_dir: Output directory
        fever_split: FEVER split to use
        max_fever: Max FEVER samples
        max_politifact: Max PolitiFact samples

    Returns:
        Statistics dict
    """
    generator = TripleGenerator()

    # Load datasets
    generator.load_fever(split=fever_split, max_samples=max_fever)
    generator.load_politifact(max_samples=max_politifact)

    # Get stats
    stats = generator.get_statistics()
    logger.info(f"Triple statistics: {stats}")

    # Export
    output_dir = Path(output_dir)
    generator.export_splits(output_dir, format="jsonl")
    generator.export_json(output_dir / "all_triples.json")

    return stats
