"""Evidence Retriever - Main RAG interface for the fact-checking pipeline.

This module provides the EvidenceRetriever class which is the primary API
for downstream components (Claim Decomposer, Stance Classifier, etc.).
"""

from dataclasses import dataclass

from ..datasets.base import EvidencePassage


@dataclass
class RetrievalResult:
    """Result from evidence retrieval.

    Attributes:
        passage: The retrieved evidence passage
        score: Similarity score (higher is more relevant)
        rank: Position in result list (1-indexed)
    """

    passage: EvidencePassage
    score: float
    rank: int


class EvidenceRetriever:
    """RAG-based evidence retriever over the indexed corpus.

    This is the main interface for retrieving evidence passages
    relevant to a given claim or query.

    Example:
        ```python
        retriever = EvidenceRetriever()
        results = retriever.retrieve("The Earth is flat", top_k=5)
        for r in results:
            print(f"{r.rank}. [{r.score:.3f}] {r.passage.text[:100]}...")
        ```
    """

    def __init__(
        self,
        index_path: str | None = None,
        embedding_model: str | None = None,
        collection_name: str | None = None,
    ):
        """Initialize the retriever.

        Args:
            index_path: Path to ChromaDB persistent storage. Uses config default if None.
            embedding_model: Sentence transformer model name. Uses config default if None.
            collection_name: ChromaDB collection name. Uses config default if None.
        """
        # Defer imports to avoid circular dependencies and heavy loading on import
        from src.config.settings import settings

        self.index_path = index_path or str(
            settings.get_absolute_path(settings.chroma_persist_dir)
        )
        self.embedding_model = embedding_model or settings.embedding_model
        self.collection_name = collection_name or settings.chroma_collection_name
        self.default_top_k = settings.default_top_k
        self.max_top_k = settings.max_top_k

        self._embedder = None
        self._collection = None
        self._initialized = False

    def _lazy_init(self) -> None:
        """Lazily initialize embedder and ChromaDB connection."""
        if self._initialized:
            return

        from sentence_transformers import SentenceTransformer
        import chromadb

        self._embedder = SentenceTransformer(self.embedding_model)
        self._client = chromadb.PersistentClient(path=self.index_path)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._initialized = True

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        dataset_filter: str | None = None,
        source_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant evidence passages for a query.

        Args:
            query: The claim or query text to search for
            top_k: Number of results to return (default from config)
            dataset_filter: Filter to specific dataset ("fever" or "politifact")
            source_filter: Filter to specific source (e.g., Wikipedia article title)

        Returns:
            List of RetrievalResult objects sorted by relevance
        """
        self._lazy_init()

        if top_k is None:
            top_k = self.default_top_k
        top_k = min(top_k, self.max_top_k)

        # Build where clause for filtering
        where = None
        if dataset_filter or source_filter:
            conditions = []
            if dataset_filter:
                conditions.append({"dataset": dataset_filter})
            if source_filter:
                conditions.append({"source": source_filter})
            where = conditions[0] if len(conditions) == 1 else {"$and": conditions}

        # Embed query
        query_embedding = self._embedder.encode(query, convert_to_numpy=True).tolist()

        # Search
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to RetrievalResult objects
        retrieval_results = []
        if results["ids"] and results["ids"][0]:
            for i, (doc_id, doc, metadata, distance) in enumerate(
                zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                # Convert cosine distance to similarity score
                score = 1 - distance

                passage = EvidencePassage(
                    id=doc_id,
                    text=doc,
                    source=metadata.get("source", ""),
                    dataset=metadata.get("dataset", ""),
                    metadata={
                        k: v
                        for k, v in metadata.items()
                        if k not in ("source", "dataset")
                    },
                )
                retrieval_results.append(
                    RetrievalResult(passage=passage, score=score, rank=i + 1)
                )

        return retrieval_results

    def retrieve_batch(
        self,
        queries: list[str],
        top_k: int | None = None,
    ) -> list[list[RetrievalResult]]:
        """Retrieve evidence for multiple queries in batch.

        More efficient than calling retrieve() multiple times.

        Args:
            queries: List of query strings
            top_k: Number of results per query

        Returns:
            List of result lists, one per query
        """
        self._lazy_init()

        if top_k is None:
            top_k = self.default_top_k
        top_k = min(top_k, self.max_top_k)

        # Batch embed
        query_embeddings = self._embedder.encode(
            queries, convert_to_numpy=True, show_progress_bar=len(queries) > 10
        ).tolist()

        # Batch search
        results = self._collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to results
        all_results = []
        for q_idx in range(len(queries)):
            query_results = []
            if results["ids"] and results["ids"][q_idx]:
                for i, (doc_id, doc, metadata, distance) in enumerate(
                    zip(
                        results["ids"][q_idx],
                        results["documents"][q_idx],
                        results["metadatas"][q_idx],
                        results["distances"][q_idx],
                    )
                ):
                    score = 1 - distance
                    passage = EvidencePassage(
                        id=doc_id,
                        text=doc,
                        source=metadata.get("source", ""),
                        dataset=metadata.get("dataset", ""),
                        metadata={
                            k: v
                            for k, v in metadata.items()
                            if k not in ("source", "dataset")
                        },
                    )
                    query_results.append(
                        RetrievalResult(passage=passage, score=score, rank=i + 1)
                    )
            all_results.append(query_results)

        return all_results

    def get_corpus_stats(self) -> dict:
        """Get statistics about the indexed corpus."""
        self._lazy_init()
        return {
            "total_passages": self._collection.count(),
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
        }
