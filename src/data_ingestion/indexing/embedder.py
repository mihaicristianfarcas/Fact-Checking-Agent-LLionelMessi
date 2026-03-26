"""Embedding generation for evidence passages.

Uses sentence-transformers to generate dense embeddings for
semantic similarity search.
"""

from typing import Iterator

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class Embedder:
    """Generate embeddings for text passages using sentence-transformers.

    Example:
        ```python
        embedder = Embedder()
        embeddings = embedder.embed(["text 1", "text 2"])
        print(embeddings.shape)  # (2, 384)
        ```
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
    ):
        """Initialize the embedder.

        Args:
            model_name: Sentence transformer model name
            device: Device to run on ("cpu", "cuda", "mps", or None for auto)
            batch_size: Batch size for embedding generation
        """
        self.model_name = model_name
        self.batch_size = batch_size

        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")

    def embed(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress and len(texts) > self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # For cosine similarity
        )
        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            numpy array of shape (dimension,)
        """
        return self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def embed_iterator(
        self,
        texts: Iterator[str],
        total: int | None = None,
    ) -> Iterator[tuple[str, np.ndarray]]:
        """Generate embeddings from an iterator, yielding (text, embedding) pairs.

        Useful for processing large datasets without loading all into memory.

        Args:
            texts: Iterator of text strings
            total: Total count for progress bar (optional)

        Yields:
            Tuples of (text, embedding)
        """
        batch = []

        for text in tqdm(texts, total=total, desc="Embedding"):
            batch.append(text)

            if len(batch) >= self.batch_size:
                embeddings = self.embed(batch, show_progress=False)
                for t, e in zip(batch, embeddings):
                    yield t, e
                batch = []

        # Process remaining
        if batch:
            embeddings = self.embed(batch, show_progress=False)
            for t, e in zip(batch, embeddings):
                yield t, e
