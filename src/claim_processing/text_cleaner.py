"""
Cleans raw evidence passage text coming from Task A's retriever before it
is consumed by the StanceClassifier.

The FEVER dataset stores Wikipedia hyperlink annotations as tab-separated
tokens appended to the actual sentence text. Example:

    RAW:
    "He had a recurring role on Fox .\tFox Broadcasting\tFox Broadcasting\t..."

    CLEAN:
    "He had a recurring role on Fox."

These annotations are meaningless to an NLI model and degrade classification
accuracy — DeBERTa tries to treat them as sentence content.
"""

from __future__ import annotations

import re


def clean_passage_text(text: str) -> str:
    """
    Strip FEVER Wikipedia annotation artifacts from a passage string.

    FEVER appends tab-separated hyperlink anchor text after the actual
    sentence, e.g.:
        "Real sentence .\tAnchor text\tAnchor text\tAnother anchor\t..."

    This function:
        1. Splits on the first tab and discards everything after it.
        2. Normalises internal whitespace.
        3. Removes any space(s) before terminal punctuation ("ruin ." → "ruin.").
        4. Ensures the sentence ends with a single full-stop if it has none.

    Args:
        text: Raw passage text as stored in ChromaDB / jsonl files.

    Returns:
        Clean, NLI-ready sentence string.
    """
    if not text:
        return text

    # Step 1 — drop Wikipedia annotation tail (everything from first \t onwards)
    clean = text.split("\t")[0]

    # Step 2 — collapse any internal runs of whitespace
    clean = re.sub(r"\s+", " ", clean).strip()

    # Step 3 — remove space(s) before terminal punctuation
    # FEVER sentences often look like "Henry was ruined ." after tab-stripping
    clean = re.sub(r"\s+([.!?])$", r"\1", clean)

    # Step 4 — ensure the sentence ends with punctuation
    if clean and clean[-1] not in ".!?":
        clean = clean + "."

    return clean


def clean_passages_in_retrieval_results(retrieval_results: list) -> list:
    """
    Clean passage text in-place for every RetrievalResult in the list.

    Because Task A's RetrievalResult/EvidencePassage are dataclasses they
    are mutable by default — we reassign r.passage.text directly and return
    the same list for convenience.

    Args:
        retrieval_results: list[RetrievalResult] from EvidenceRetriever.

    Returns:
        The same list with each r.passage.text cleaned.
    """
    for r in retrieval_results:
        r.passage.text = clean_passage_text(r.passage.text)
    return retrieval_results