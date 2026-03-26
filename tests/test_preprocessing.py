"""Tests for text preprocessing utilities."""

import pytest

from src.data_ingestion.preprocessing import (
    TextChunker,
    TextCleaner,
    SentenceSplitter,
    chunk_text,
    clean_text,
    split_sentences,
)


class TestTextCleaner:
    """Test TextCleaner class."""

    def test_basic_cleaning(self):
        cleaner = TextCleaner()
        result = cleaner.clean("  Hello   World  ")
        assert result == "Hello World"

    def test_html_removal(self):
        cleaner = TextCleaner(remove_html=True)
        result = cleaner.clean("<p>Hello <b>World</b></p>")
        assert "Hello" in result
        assert "World" in result
        assert "<" not in result

    def test_whitespace_normalization(self):
        cleaner = TextCleaner()
        result = cleaner.clean("Hello\t\n  World")
        assert result == "Hello World"

    def test_lowercase(self):
        cleaner = TextCleaner(lowercase=True)
        result = cleaner.clean("Hello World")
        assert result == "hello world"

    def test_empty_input(self):
        cleaner = TextCleaner()
        assert cleaner.clean("") == ""
        assert cleaner.clean(None) == ""


class TestTextChunker:
    """Test TextChunker class."""

    def test_basic_chunking(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "A" * 100
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        assert all(len(c.text) <= 50 for c in chunks)

    def test_short_text(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        text = "Short text"
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0].text == "Short text"

    def test_chunk_indices(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "A" * 100
        chunks = chunker.chunk(text)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_empty_input(self):
        chunker = TextChunker()
        assert chunker.chunk("") == []


class TestSentenceSplitter:
    """Test SentenceSplitter class."""

    def test_basic_splitting(self):
        splitter = SentenceSplitter()
        text = "First sentence. Second sentence. Third sentence."
        sentences = splitter.split(text)
        assert len(sentences) == 3

    def test_different_punctuation(self):
        splitter = SentenceSplitter(min_length=5)
        text = "Question? Exclamation! Statement."
        sentences = splitter.split(text)
        assert len(sentences) >= 2

    def test_min_length_filter(self):
        splitter = SentenceSplitter(min_length=20)
        text = "Hi. This is a longer sentence that should pass."
        sentences = splitter.split(text)
        # "Hi." is too short, should be filtered
        assert all(len(s) >= 20 for s in sentences)

    def test_empty_input(self):
        splitter = SentenceSplitter()
        assert splitter.split("") == []


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_clean_text(self):
        result = clean_text("  <p>Hello</p>  ")
        assert "Hello" in result
        assert "<" not in result

    def test_chunk_text(self):
        chunks = chunk_text("A" * 100, chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 1

    def test_split_sentences(self):
        sentences = split_sentences("One. Two. Three.", min_length=3)
        assert len(sentences) >= 2
