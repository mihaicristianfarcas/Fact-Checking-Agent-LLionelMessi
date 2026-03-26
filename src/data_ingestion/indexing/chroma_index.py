"""ChromaDB indexing operations.

Handles creating, updating, and querying the vector index
for evidence retrieval.
"""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger
from tqdm import tqdm

from ..datasets.base import EvidencePassage
from .embedder import Embedder


class ChromaIndex:
    """ChromaDB-based vector index for evidence passages.

    Example:
        ```python
        index = ChromaIndex(persist_dir="data/index/chroma")

        # Add passages
        passages = [EvidencePassage(id="1", text="text", source="wiki", dataset="fever")]
        index.add_passages(passages)

        # Search
        results = index.search("query text", top_k=5)
        ```
    """

    def __init__(
        self,
        persist_dir: str | Path = "data/index/chroma",
        collection_name: str = "evidence_corpus",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 100,
    ):
        """Initialize the ChromaDB index.

        Args:
            persist_dir: Directory for persistent storage
            collection_name: Name of the ChromaDB collection
            embedding_model: Sentence transformer model for embeddings
            batch_size: Batch size for adding documents
        """
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.batch_size = batch_size

        # Create persist directory
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize embedder
        self.embedder = Embedder(model_name=embedding_model, batch_size=batch_size)

        # Initialize ChromaDB client
        logger.info(f"Initializing ChromaDB at {self.persist_dir}")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"Collection '{collection_name}' initialized with {self.collection.count()} documents"
        )

    def add_passages(
        self,
        passages: list[EvidencePassage],
        show_progress: bool = True,
    ) -> int:
        """Add passages to the index.

        Args:
            passages: List of EvidencePassage objects to add
            show_progress: Whether to show progress bar

        Returns:
            Number of passages added
        """
        if not passages:
            return 0

        added = 0
        batches = [
            passages[i : i + self.batch_size]
            for i in range(0, len(passages), self.batch_size)
        ]

        iterator = tqdm(batches, desc="Indexing") if show_progress else batches

        for batch in iterator:
            ids = [p.id for p in batch]
            texts = [p.text for p in batch]
            metadatas = [
                {
                    "source": p.source,
                    "dataset": p.dataset,
                    **{k: str(v) for k, v in p.metadata.items()},
                }
                for p in batch
            ]

            # Generate embeddings
            embeddings = self.embedder.embed(texts, show_progress=False)

            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
            )
            added += len(batch)

        logger.info(f"Added {added} passages to index")
        return added

    def add_passages_from_iterator(
        self,
        passages,
        total: int | None = None,
    ) -> int:
        """Add passages from an iterator (memory efficient for large datasets).

        Args:
            passages: Iterator of EvidencePassage objects
            total: Total count for progress bar (optional)

        Returns:
            Number of passages added
        """
        batch = []
        added = 0

        pbar = tqdm(passages, total=total, desc="Indexing")
        for passage in pbar:
            batch.append(passage)

            if len(batch) >= self.batch_size:
                added += self.add_passages(batch, show_progress=False)
                batch = []

        # Process remaining
        if batch:
            added += self.add_passages(batch, show_progress=False)

        return added

    def search(
        self,
        query: str,
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Search for relevant passages.

        Args:
            query: Query text
            top_k: Number of results to return
            where: Metadata filter (ChromaDB where clause)

        Returns:
            List of result dicts with id, text, metadata, score
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_single(query).tolist()

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted = []
        if results["ids"] and results["ids"][0]:
            for i, (doc_id, doc, metadata, distance) in enumerate(
                zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                formatted.append(
                    {
                        "id": doc_id,
                        "text": doc,
                        "metadata": metadata,
                        "score": 1 - distance,  # Convert distance to similarity
                        "rank": i + 1,
                    }
                )

        return formatted

    def delete_by_dataset(self, dataset: str) -> int:
        """Delete all passages from a specific dataset.

        Args:
            dataset: Dataset name to delete ("fever" or "politifact")

        Returns:
            Number of documents deleted
        """
        # Get IDs to delete
        results = self.collection.get(
            where={"dataset": dataset},
            include=[],
        )

        if not results["ids"]:
            return 0

        count = len(results["ids"])
        self.collection.delete(ids=results["ids"])
        logger.info(f"Deleted {count} passages from dataset '{dataset}'")
        return count

    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Cleared all documents from index")

    def get_stats(self) -> dict:
        """Get index statistics."""
        count = self.collection.count()

        # Sample to get dataset distribution
        if count > 0:
            sample = self.collection.get(
                limit=min(count, 1000),
                include=["metadatas"],
            )
            dataset_counts: dict[str, int] = {}
            for meta in sample["metadatas"]:
                ds = meta.get("dataset", "unknown")
                dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
        else:
            dataset_counts = {}

        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "persist_dir": str(self.persist_dir),
            "embedding_model": self.embedder.model_name,
            "embedding_dimension": self.embedder.dimension,
            "dataset_distribution": dataset_counts,
        }
