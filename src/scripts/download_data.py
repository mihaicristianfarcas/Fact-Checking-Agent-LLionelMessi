#!/usr/bin/env python3
"""Download datasets for the fact-checking pipeline.

Usage:
    python -m src.scripts.download_data [--fever-split train] [--politifact-max 5000]
"""

import argparse

from loguru import logger

from src.data_ingestion.datasets import FeverDataset, PolitifactDataset


def main():
    parser = argparse.ArgumentParser(description="Download fact-checking datasets")
    parser.add_argument(
        "--fever-split",
        type=str,
        default="train",
        choices=["train", "labelled_dev", "paper_dev", "paper_test"],
        help="FEVER split to download",
    )
    parser.add_argument(
        "--politifact-max",
        type=int,
        default=None,
        help="Maximum PolitiFact samples to load",
    )
    parser.add_argument(
        "--load-wiki",
        action="store_true",
        help="Also load FEVER Wikipedia pages (large download)",
    )
    parser.add_argument(
        "--wiki-limit",
        type=int,
        default=None,
        help="Limit Wikipedia pages to load (for testing)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Downloading fact-checking datasets")
    logger.info("=" * 60)

    # Download FEVER
    logger.info("\n[1/2] Loading FEVER dataset...")
    fever = FeverDataset(split=args.fever_split)
    fever.load()
    fever_stats = fever.get_statistics()
    logger.info(f"FEVER stats: {fever_stats}")

    if args.load_wiki:
        logger.info("\nLoading FEVER Wikipedia pages...")
        fever.load_wiki_pages(limit=args.wiki_limit)
        logger.info(f"Wiki pages loaded: {fever.get_statistics()['wiki_pages_loaded']}")

    # Download PolitiFact/LIAR
    logger.info("\n[2/2] Loading PolitiFact/LIAR dataset...")
    for split in ["train", "validation", "test"]:
        pf = PolitifactDataset(split=split, max_samples=args.politifact_max)
        pf.load()
        pf_stats = pf.get_statistics()
        logger.info(f"PolitiFact {split} stats: {pf_stats}")

    logger.info("\n" + "=" * 60)
    logger.info("Dataset download complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
