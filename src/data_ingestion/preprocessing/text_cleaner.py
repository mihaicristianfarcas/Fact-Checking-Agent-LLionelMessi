"""Text preprocessing utilities for the fact-checking pipeline.

Provides text cleaning, normalization, and chunking for RAG indexing.
"""

import re
import unicodedata
from dataclasses import dataclass

from bs4 import BeautifulSoup
from ftfy import fix_text


@dataclass
class TextChunk:
    """A chunk of text for indexing."""

    text: str
    start_char: int
    end_char: int
    chunk_index: int


class TextCleaner:
    """Clean and normalize text for consistent processing.

    Example:
        ```python
        cleaner = TextCleaner()
        clean = cleaner.clean("Some  messy   text<br>with HTML")
        # Returns: "Some messy text with HTML"
        ```
    """

    def __init__(
        self,
        lowercase: bool = False,
        remove_html: bool = True,
        normalize_unicode: bool = True,
        normalize_whitespace: bool = True,
        fix_encoding: bool = True,
    ):
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.normalize_unicode = normalize_unicode
        self.normalize_whitespace = normalize_whitespace
        self.fix_encoding = fix_encoding

    def clean(self, text: str) -> str:
        """Apply all cleaning steps to text."""
        if not text:
            return ""

        # Fix encoding issues
        if self.fix_encoding:
            text = fix_text(text)

        # Remove HTML tags
        if self.remove_html:
            text = self._remove_html(text)

        # Normalize unicode
        if self.normalize_unicode:
            text = self._normalize_unicode(text)

        # Normalize whitespace
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)

        # Lowercase
        if self.lowercase:
            text = text.lower()

        return text.strip()

    def _remove_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        soup = BeautifulSoup(text, "lxml")
        return soup.get_text(separator=" ")

    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters to NFC form."""
        return unicodedata.normalize("NFC", text)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace: collapse multiple spaces, convert tabs/newlines."""
        text = re.sub(r"[\t\n\r\f\v]+", " ", text)
        text = re.sub(r" +", " ", text)
        return text


class TextChunker:
    """Split text into overlapping chunks for RAG indexing.

    Example:
        ```python
        chunker = TextChunker(chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk("Long document text...")
        for chunk in chunks:
            print(f"Chunk {chunk.chunk_index}: {chunk.text[:50]}...")
        ```
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        length_function: callable = len,
    ):
        """Initialize the chunker.

        Args:
            chunk_size: Target size of each chunk (in characters or tokens)
            chunk_overlap: Number of characters/tokens to overlap between chunks
            length_function: Function to measure text length (default: len)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

    def chunk(self, text: str) -> list[TextChunk]:
        """Split text into overlapping chunks."""
        if not text:
            return []

        # Simple character-based chunking with sentence awareness
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size

            # If not at the end, try to break at sentence boundary
            if end < len(text):
                # Look for sentence boundary within the last 20% of chunk
                search_start = start + int(self.chunk_size * 0.8)
                search_text = text[search_start:end]

                # Find last sentence boundary
                for boundary in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                    last_boundary = search_text.rfind(boundary)
                    if last_boundary != -1:
                        end = search_start + last_boundary + len(boundary)
                        break

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        start_char=start,
                        end_char=end,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

            # Move start, accounting for overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break

        return chunks


class SentenceSplitter:
    """Split text into sentences for passage-level indexing.

    Example:
        ```python
        splitter = SentenceSplitter()
        sentences = splitter.split("First sentence. Second sentence! Third?")
        # Returns: ["First sentence.", "Second sentence!", "Third?"]
        ```
    """

    def __init__(self, min_length: int = 10):
        """Initialize the splitter.

        Args:
            min_length: Minimum sentence length to include
        """
        self.min_length = min_length
        # Simple sentence boundary pattern
        self._pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    def split(self, text: str) -> list[str]:
        """Split text into sentences."""
        if not text:
            return []

        sentences = self._pattern.split(text)
        return [s.strip() for s in sentences if len(s.strip()) >= self.min_length]


# Convenience functions
def clean_text(text: str, **kwargs) -> str:
    """Clean text with default settings."""
    return TextCleaner(**kwargs).clean(text)


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[TextChunk]:
    """Chunk text with specified parameters."""
    return TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap).chunk(text)


def split_sentences(text: str, min_length: int = 10) -> list[str]:
    """Split text into sentences."""
    return SentenceSplitter(min_length=min_length).split(text)
