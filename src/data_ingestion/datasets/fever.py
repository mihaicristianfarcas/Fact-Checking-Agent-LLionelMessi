"""FEVER dataset loader.

FEVER (Fact Extraction and VERification) is a dataset of 185,445 claims
generated from Wikipedia. Each claim is labeled as SUPPORTED, REFUTED,
or NOT ENOUGH INFO based on evidence from Wikipedia articles.

HuggingFace schema note (fever/fever, v1.0):
  The dataset is stored flat — one row per evidence sentence, not one row per
  claim.  Multiple rows share the same claim `id`.  Fields per row:
    id, label, claim,
    evidence_annotation_id, evidence_id, evidence_wiki_url, evidence_sentence_id
  NOT ENOUGH INFO rows have evidence_wiki_url='' and evidence_sentence_id=-1.
"""

from collections import defaultdict
from collections.abc import Iterator
from typing import Any

from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from .base import BaseDataset, Claim, EvidencePassage, Verdict


FEVER_LABEL_MAP = {
    "SUPPORTS": Verdict.SUPPORTED,
    "REFUTES": Verdict.REFUTED,
    "NOT ENOUGH INFO": Verdict.NOT_ENOUGH_INFO,
}


class FeverDataset(BaseDataset):
    """Loader for the FEVER dataset via HuggingFace.

    Example:
        ```python
        dataset = FeverDataset(split="train")
        dataset.load()

        for claim in dataset.iter_claims():
            print(f"{claim.verdict}: {claim.text}")
        ```
    """

    def __init__(self, split: str = "train", cache_dir: str | None = None):
        """Initialize the FEVER dataset loader.

        Args:
            split: Dataset split to load ("train", "labelled_dev", "paper_dev", "paper_test")
            cache_dir: Directory to cache downloaded data
        """
        self.split = split
        self.cache_dir = cache_dir
        self._dataset = None
        self._wiki_pages: dict[str, str] = {}
        # Grouped view: claim_id -> {"claim", "label", "evidence_records"}
        self._claims_grouped: dict[int, dict] = {}

    @property
    def name(self) -> str:
        return "fever"

    def load(self) -> None:
        """Load the FEVER dataset from HuggingFace."""
        logger.info(f"Loading FEVER dataset (split={self.split})...")

        # Use the namespaced "fever/fever" identifier (canonical name).
        # verification_mode="no_checks" bypasses the NonMatchingChecksumError
        # caused by FEVER's download URLs changing on fever.ai in 2022 while
        # the HuggingFace script still carries old checksums.
        self._dataset = load_dataset(
            "fever/fever",
            "v1.0",
            split=self.split,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            verification_mode="no_checks",
        )

        # Group flat rows by claim id so iter_claims() yields one Claim per
        # unique claim, collecting all its evidence sentences.
        self._claims_grouped = {}
        for row in self._dataset:
            cid = row["id"]
            if cid not in self._claims_grouped:
                self._claims_grouped[cid] = {
                    "claim": row["claim"],
                    "label": row["label"],
                    "evidence_records": [],
                }
            # Only keep rows that carry real evidence (wiki_url is non-empty)
            if row["evidence_wiki_url"]:
                self._claims_grouped[cid]["evidence_records"].append({
                    "wiki_url": row["evidence_wiki_url"],
                    "sentence_id": row["evidence_sentence_id"],
                    "annotation_id": row["evidence_annotation_id"],
                    "evidence_id": row["evidence_id"],
                })

        logger.info(
            f"Loaded {len(self._claims_grouped)} claims from FEVER {self.split} "
            f"({len(self._dataset)} total rows)"
        )

    def load_wiki_pages(self, limit: int | None = None) -> None:
        """Load Wikipedia pages for evidence lookup.

        Args:
            limit: Maximum number of pages to load (for testing)
        """
        logger.info("Loading FEVER Wikipedia pages...")

        wiki_dataset = load_dataset(
            "fever/fever",
            "wiki_pages",
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            verification_mode="no_checks",
        )

        count = 0
        for page in tqdm(wiki_dataset["wikipedia_pages"], desc="Loading wiki pages"):
            page_id = page["id"]
            lines = page.get("lines", "")
            if page_id and lines:
                self._wiki_pages[page_id] = lines
            count += 1
            if limit and count >= limit:
                break

        logger.info(f"Loaded {len(self._wiki_pages)} Wikipedia pages")

    def _get_sentence_text(self, wiki_page: str, sentence_id: int) -> str:
        """Look up a specific sentence from a loaded Wikipedia page.

        The `lines` field uses the format "sentence_id\\ttext" per line.
        """
        if wiki_page not in self._wiki_pages:
            return ""

        for line in self._wiki_pages[wiki_page].split("\n"):
            if "\t" not in line:
                continue
            sid_str, text = line.split("\t", 1)
            try:
                if int(sid_str) == sentence_id:
                    return text
            except ValueError:
                continue
        return ""

    def _parse_evidence(self, evidence_records: list[dict]) -> list[EvidencePassage]:
        """Build EvidencePassage objects from flat evidence records.

        Args:
            evidence_records: List of dicts with keys wiki_url, sentence_id,
                annotation_id, evidence_id.
        """
        passages = []
        seen = set()

        for record in evidence_records:
            wiki_page = record["wiki_url"]
            sentence_id = record["sentence_id"]

            if not wiki_page or sentence_id < 0:
                continue

            passage_id = f"fever_{wiki_page}_{sentence_id}"
            if passage_id in seen:
                continue
            seen.add(passage_id)

            text = self._get_sentence_text(wiki_page, sentence_id)

            passages.append(
                EvidencePassage(
                    id=passage_id,
                    text=text,
                    source=wiki_page,
                    dataset="fever",
                    metadata={"sentence_id": sentence_id},
                )
            )

        return passages

    def iter_claims(self) -> Iterator[Claim]:
        """Iterate over all claims in the dataset (one Claim per unique claim id)."""
        if self._claims_grouped is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        for cid, data in self._claims_grouped.items():
            verdict = FEVER_LABEL_MAP.get(data["label"])
            evidence = self._parse_evidence(data["evidence_records"])

            yield Claim(
                id=f"fever_{cid}",
                text=data["claim"],
                verdict=verdict,
                evidence=evidence,
                dataset="fever",
                metadata={
                    "original_id": cid,
                    "original_label": data["label"],
                },
            )

    def iter_evidence(self) -> Iterator[EvidencePassage]:
        """Iterate over all evidence passages (Wikipedia sentences)."""
        if not self._wiki_pages:
            logger.warning("Wiki pages not loaded. Call load_wiki_pages() first.")
            return

        for page_id, content in self._wiki_pages.items():
            for line in content.split("\n"):
                if not line.strip():
                    continue

                if "\t" in line:
                    sid_str, text = line.split("\t", 1)
                    try:
                        sentence_id = int(sid_str)
                    except ValueError:
                        continue
                else:
                    text = line
                    sentence_id = 0

                if not text.strip():
                    continue

                yield EvidencePassage(
                    id=f"fever_{page_id}_{sentence_id}",
                    text=text,
                    source=page_id,
                    dataset="fever",
                    metadata={"sentence_id": sentence_id},
                )

    def get_statistics(self) -> dict:
        """Return dataset statistics."""
        if not self._claims_grouped:
            return {"loaded": False}

        label_counts: dict[str, int] = {}
        for data in self._claims_grouped.values():
            label = data["label"]
            label_counts[label] = label_counts.get(label, 0) + 1

        return {
            "loaded": True,
            "split": self.split,
            "total_claims": len(self._claims_grouped),
            "label_distribution": label_counts,
            "wiki_pages_loaded": len(self._wiki_pages),
        }
