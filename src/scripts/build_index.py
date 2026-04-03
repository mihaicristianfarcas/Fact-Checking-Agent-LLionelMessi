#!/usr/bin/env python3
"""Build the ChromaDB index from downloaded datasets.

Usage:
    python -m src.scripts.build_index [--fever-split train] [--clear]
    python -m src.scripts.build_index --political-filter --clear
"""

import argparse
from pathlib import Path

from loguru import logger

from src.config import settings
from src.data_ingestion.datasets import FeverDataset, PolitifactDataset
from src.data_ingestion.datasets.base import EvidencePassage
from src.data_ingestion.indexing import ChromaIndex
from src.data_ingestion.preprocessing import clean_text, chunk_text


# Keywords matched against Wikipedia page titles (lowercased, underscores kept).
# A page is included if its title contains any of these substrings.
POLITICAL_KEYWORDS = {
    # Roles & titles
    "senator", "president", "governor", "congressman", "representative",
    "minister", "secretary", "mayor", "candidate", "politician", "chancellor",
    "ambassador", "legislator", "assemblyman", "assemblywoman", "councillor",
    # Institutions
    "congress", "senate", "parliament", "government", "administration",
    "department", "committee", "cabinet", "embassy", "legislature",
    "judiciary", "supreme_court", "white_house", "pentagon",
    # Processes & events
    "election", "campaign", "vote", "voting", "bill", "legislation",
    "amendment", "constitution", "referendum", "impeachment", "inauguration",
    "primary", "caucus", "debate", "filibuster",
    # Parties & ideologies
    "democrat", "republican", "liberal", "conservative", "socialist",
    "political_party", "libertarian",
    # Policy topics
    "immigration", "healthcare", "taxation", "budget", "deficit", "debt",
    "sanctions", "treaty", "diplomacy", "foreign_policy", "trade",
    "climate", "gun_control", "abortion", "welfare",
    # Security & military
    "war", "military", "terrorism", "national_security", "cia", "fbi",
    "nsa", "intelligence", "drone", "nuclear",
    # General political
    "political", "united_states", "washington",
}


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


def index_fever_wiki_political(index: ChromaIndex, limit: int | None = None) -> int:
    """Index Wikipedia pages whose titles match political keywords.

    Streams the full FEVER wiki dump without loading it all into memory,
    filters to politically relevant pages, then chunks each page into
    ~500-character passages and indexes them.

    This produces a focused but broad evidence corpus suitable for RAG
    over political claims at inference time.
    """
    from datasets import load_dataset
    from tqdm import tqdm

    logger.info("Streaming FEVER Wikipedia pages (political filter)...")
    wiki_dataset = load_dataset(
        "fever/fever",
        "wiki_pages",
        trust_remote_code=True,
        verification_mode="no_checks",
    )

    passages = []
    pages_matched = 0
    total_passages = 0

    for page in tqdm(wiki_dataset["wikipedia_pages"], desc="Filtering wiki pages"):
        page_id = page.get("id", "")
        if not page_id:
            continue

        # Filter: page title must contain at least one political keyword
        title_lower = page_id.lower()
        if not any(kw in title_lower for kw in POLITICAL_KEYWORDS):
            continue

        lines = page.get("lines", "")
        if not lines:
            continue

        # Reconstruct full page text from sentence lines ("sid\ttext")
        sentences = []
        for line in lines.split("\n"):
            if "\t" not in line:
                continue
            _, text = line.split("\t", 1)
            text = text.strip()
            if text:
                sentences.append(text)

        if not sentences:
            continue

        full_text = clean_text(" ".join(sentences))
        if not full_text:
            continue

        # Chunk the page into ~500-char overlapping passages
        for chunk in chunk_text(full_text, chunk_size=500, chunk_overlap=50):
            passages.append(
                EvidencePassage(
                    id=f"fever_{page_id}_chunk_{chunk.chunk_index}",
                    text=chunk.text,
                    source=page_id,
                    dataset="fever",
                    metadata={"chunk_index": chunk.chunk_index},
                )
            )

        pages_matched += 1

        # Flush to index in batches to keep memory low
        if len(passages) >= 5000:
            index.add_passages(passages)
            total_passages += len(passages)
            passages = []

        if limit and pages_matched >= limit:
            break

    if passages:
        index.add_passages(passages)
        total_passages += len(passages)

    logger.info(
        f"Political filter: {pages_matched} pages matched, "
        f"{total_passages} passages indexed"
    )
    return total_passages


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
        help="Index all Wikipedia sentences (not just claim evidence) — 25M+ passages, impractical",
    )
    parser.add_argument(
        "--political-filter",
        action="store_true",
        help="Index only Wikipedia pages with politically relevant titles (recommended for political fact-checking)",
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
        if args.political_filter:
            added = index_fever_wiki_political(index, limit=args.wiki_limit)
        elif args.fever_full_wiki:
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
