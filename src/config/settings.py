"""Configuration management for the Data & Ingestion pipeline."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FACTCHECK_",
        case_sensitive=False,
    )

    # Paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent
    )
    data_dir: Path = Field(default=Path("data"))
    raw_data_dir: Path = Field(default=Path("data/raw"))
    processed_data_dir: Path = Field(default=Path("data/processed"))
    index_dir: Path = Field(default=Path("data/index"))

    # Embedding model
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_dimension: int = Field(default=384)
    embedding_batch_size: int = Field(default=32)

    # ChromaDB
    chroma_persist_dir: Path = Field(default=Path("data/index/chroma"))
    chroma_collection_name: str = Field(default="evidence_corpus")

    # Retrieval
    default_top_k: int = Field(default=10)
    max_top_k: int = Field(default=100)

    # Dataset settings
    fever_split: Literal["train", "dev", "test"] = Field(default="train")
    politifact_max_samples: int = Field(default=5000)

    # Processing
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)

    # Stance Classifier
    stance_confidence_threshold: float = Field(default=0.65)

    # Logging
    log_level: str = Field(default="INFO")

    def get_absolute_path(self, relative_path: Path) -> Path:
        """Convert relative path to absolute path from project root."""
        if relative_path.is_absolute():
            return relative_path
        return self.project_root / relative_path


# Global settings instance
settings = Settings()
