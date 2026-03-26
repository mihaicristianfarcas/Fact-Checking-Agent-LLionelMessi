"""Data Ingestion module for Fact-Checking Agent.

This module provides:
- Dataset loaders for FEVER and PolitiFact
- Text preprocessing utilities
- ChromaDB indexing pipeline
- Evidence retriever (RAG)
- Ground-truth triples generator
"""

from .datasets.base import Claim, EvidencePassage, Verdict
from .retriever.evidence_retriever import EvidenceRetriever

__all__ = [
    "Claim",
    "EvidencePassage",
    "Verdict",
    "EvidenceRetriever",
]
