#!/usr/bin/env python3
"""Validate the evidence corpus and index.

Usage:
    python -m src.scripts.validate_corpus [--sample-queries 10]
"""

import argparse
import random

from loguru import logger

from src.config import settings
from src.data_ingestion import EvidenceRetriever
from src.data_ingestion.datasets import FeverDataset


def validate_index_stats(retriever: EvidenceRetriever) -> bool:
    """Check basic index statistics."""
    logger.info("Checking index statistics...")
    stats = retriever.get_corpus_stats()

    logger.info(f"  Total passages: {stats['total_passages']}")
    logger.info(f"  Collection: {stats['collection_name']}")
    logger.info(f"  Embedding model: {stats['embedding_model']}")

    if stats["total_passages"] == 0:
        logger.error("Index is empty!")
        return False

    return True


def validate_retrieval(retriever: EvidenceRetriever, num_queries: int = 10) -> dict:
    """Run sample queries and validate retrieval quality."""
    logger.info(f"Running {num_queries} sample queries...")

    # Load some FEVER claims for testing
    logger.info("Loading FEVER dev set for test queries...")
    try:
        dataset = FeverDataset(split="labelled_dev")
        dataset.load()
        claims = list(dataset.iter_claims())[:100]
    except Exception as e:
        logger.warning(f"Could not load FEVER dev set: {e}")
        # Use hardcoded test queries
        claims = None

    if claims:
        test_queries = random.sample(claims, min(num_queries, len(claims)))
        query_texts = [c.text for c in test_queries]
    else:
        # Fallback test queries
        query_texts = [
            "The Earth is approximately 4.5 billion years old.",
            "Barack Obama was the 44th President of the United States.",
            "The Great Wall of China is visible from space.",
            "Water boils at 100 degrees Celsius at sea level.",
            "Albert Einstein developed the theory of relativity.",
        ][:num_queries]

    results_summary = {
        "total_queries": len(query_texts),
        "queries_with_results": 0,
        "avg_top_score": 0.0,
        "sample_results": [],
    }

    total_top_score = 0.0

    for i, query in enumerate(query_texts):
        results = retriever.retrieve(query, top_k=5)

        if results:
            results_summary["queries_with_results"] += 1
            top_score = results[0].score
            total_top_score += top_score

            # Log first few
            if i < 3:
                logger.info(f"\nQuery: {query[:80]}...")
                logger.info(f"  Top result (score={top_score:.3f}): {results[0].passage.text[:100]}...")

            results_summary["sample_results"].append({
                "query": query[:100],
                "top_score": top_score,
                "num_results": len(results),
            })
        else:
            logger.warning(f"No results for query: {query[:50]}...")

    if results_summary["queries_with_results"] > 0:
        results_summary["avg_top_score"] = (
            total_top_score / results_summary["queries_with_results"]
        )

    return results_summary


def main():
    parser = argparse.ArgumentParser(description="Validate evidence corpus")
    parser.add_argument(
        "--sample-queries",
        type=int,
        default=10,
        help="Number of sample queries to run",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Index directory (default from config)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Validating Evidence Corpus")
    logger.info("=" * 60)

    # Initialize retriever
    index_dir = args.index_dir or str(
        settings.get_absolute_path(settings.chroma_persist_dir)
    )
    retriever = EvidenceRetriever(index_path=index_dir)

    # Run validations
    all_passed = True

    # 1. Check index stats
    if not validate_index_stats(retriever):
        all_passed = False

    # 2. Test retrieval
    results = validate_retrieval(retriever, num_queries=args.sample_queries)
    logger.info("\nRetrieval validation results:")
    logger.info(f"  Queries with results: {results['queries_with_results']}/{results['total_queries']}")
    logger.info(f"  Average top score: {results['avg_top_score']:.3f}")

    if results["queries_with_results"] < results["total_queries"] * 0.8:
        logger.warning("Less than 80% of queries returned results")

    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("✓ Validation PASSED")
    else:
        logger.error("✗ Validation FAILED")
    logger.info("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
