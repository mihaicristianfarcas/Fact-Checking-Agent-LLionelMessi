"""Text-formatting helpers shared across components."""

from __future__ import annotations


def display_fever_source(source: str | None) -> str:
    """Unmangle FEVER's pre-tokenized title tokens for human-readable display.

    FEVER stores page titles with bracket tokens (-LRB-, -RRB-, -LSB-, -RSB-)
    and underscores instead of spaces. This restores the original form.
    """
    return (
        (source or "unknown")
        .replace("_", " ")
        .replace("-LRB-", "(")
        .replace("-RRB-", ")")
        .replace("-LSB-", "[")
        .replace("-RSB-", "]")
    )
