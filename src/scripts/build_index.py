#!/usr/bin/env python3
"""Build the ChromaDB index from downloaded datasets.

Usage:
    python -m src.scripts.build_index [--fever-split train] [--clear]
"""

import argparse
from pathlib import Path

from loguru import logger

from src.config import settings
from src.data_ingestion.datasets import FeverDataset, PolitifactDataset
from src.data_ingestion.datasets.base import EvidencePassage
from src.data_ingestion.indexing import ChromaIndex
from src.data_ingestion.preprocessing import clean_text


def index_fever(index: ChromaIndex, split: str = "train") -> int:
    """Index FEVER dataset evidence passages.

    For FEVER, we index the Wikipedia evidence sentences that are
    linked to claims. This requires loading the wiki_pages.
    """
    logger.info(f"Loading FEVER {split} for indexing...")
    dataset = FeverDataset(split=split)
    dataset.load()
    dataset.load_wiki_pages()

    # Collect unique evidence passages from claims
    logger.info("Extracting evidence passages from claims...")
    passages = []
    seen_ids = set()

    for claim in dataset.iter_claims():
        for evidence in claim.evidence:
            if evidence.id not in seen_ids and evidence.text.strip():
                # Clean the text
                evidence.text = clean_text(evidence.text)
                if evidence.text:
                    passages.append(evidence)
                    seen_ids.add(evidence.id)

    logger.info(f"Found {len(passages)} unique evidence passages")

    # Add to index
    added = index.add_passages(passages)
    return added


def index_fever_wiki(index: ChromaIndex, limit: int | None = None) -> int:
    """Index all FEVER Wikipedia sentences (more comprehensive).

    This indexes all Wikipedia sentences, not just those linked to claims.
    Use this for better retrieval coverage but larger index size.
    """
    logger.info("Loading FEVER Wikipedia pages for full indexing...")
    dataset = FeverDataset(split="train")
    dataset.load_wiki_pages(limit=limit)

    # Index all evidence from wiki pages
    passages = list(dataset.iter_evidence())
    logger.info(f"Found {len(passages)} Wikipedia sentences to index")

    # Clean texts
    for p in passages:
        p.text = clean_text(p.text)

    # Filter empty
    passages = [p for p in passages if p.text.strip()]
    logger.info(f"After cleaning: {len(passages)} passages")

    added = index.add_passages(passages)
    return added


def index_politifact(index: ChromaIndex, max_samples: int | None = None) -> int:
    """Index PolitiFact claims as evidence passages.

    Note: LIAR dataset doesn't include evidence text, so we index the
    claims themselves. These serve as examples of fact-checked statements.
    """
    logger.info("Loading PolitiFact/LIAR for indexing...")

    passages = []
    for split in ["train", "validation", "test"]:
        dataset = PolitifactDataset(split=split, max_samples=max_samples)
        dataset.load()

        for claim in dataset.iter_claims():
            # Create evidence passage from the claim text
            passage = EvidencePassage(
                id=claim.id,
                text=clean_text(claim.text),
                source="politifact",
                dataset="politifact",
                metadata={
                    "verdict": claim.verdict.value if claim.verdict else "",
                    "speaker": claim.metadata.get("speaker", ""),
                    "context": claim.metadata.get("context", ""),
                },
            )
            if passage.text.strip():
                passages.append(passage)

    logger.info(f"Found {len(passages)} PolitiFact claims to index")

    added = index.add_passages(passages)
    return added


def main():
    parser = argparse.ArgumentParser(description="Build evidence index")
    parser.add_argument(
        "--fever-split",
        type=str,
        default="train",
        help="FEVER split to index",
    )
    parser.add_argument(
        "--fever-full-wiki",
        action="store_true",
        help="Index all Wikipedia sentences (not just claim evidence)",
    )
    parser.add_argument(
        "--wiki-limit",
        type=int,
        default=None,
        help="Limit Wikipedia pages for testing",
    )
    parser.add_argument(
        "--politifact-max",
        type=int,
        default=None,
        help="Maximum PolitiFact samples",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Index directory (default from config)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing index before building",
    )
    parser.add_argument(
        "--skip-fever",
        action="store_true",
        help="Skip FEVER indexing",
    )
    parser.add_argument(
        "--skip-politifact",
        action="store_true",
        help="Skip PolitiFact indexing",
    )
    args = parser.parse_args()

    # Initialize index
    index_dir = args.index_dir or str(
        settings.get_absolute_path(settings.chroma_persist_dir)
    )
    logger.info(f"Index directory: {index_dir}")

    index = ChromaIndex(
        persist_dir=index_dir,
        collection_name=settings.chroma_collection_name,
        embedding_model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )

    if args.clear:
        logger.info("Clearing existing index...")
        index.clear()

    logger.info("=" * 60)
    logger.info("Building evidence index")
    logger.info("=" * 60)

    total_added = 0

    # Index FEVER
    if not args.skip_fever:
        logger.info("\n[1/2] Indexing FEVER dataset...")
        if args.fever_full_wiki:
            added = index_fever_wiki(index, limit=args.wiki_limit)
        else:
            added = index_fever(index, split=args.fever_split)
        total_added += added
        logger.info(f"FEVER: Added {added} passages")

    # Index PolitiFact
    if not args.skip_politifact:
        logger.info("\n[2/2] Indexing PolitiFact dataset...")
        added = index_politifact(index, max_samples=args.politifact_max)
        total_added += added
        logger.info(f"PolitiFact: Added {added} passages")

    # Print final stats
    logger.info("\n" + "=" * 60)
    logger.info("Index build complete!")
    logger.info("=" * 60)
    stats = index.get_stats()
    logger.info(f"Total documents: {stats['total_documents']}")
    logger.info(f"Dataset distribution: {stats['dataset_distribution']}")


if __name__ == "__main__":
    main()
