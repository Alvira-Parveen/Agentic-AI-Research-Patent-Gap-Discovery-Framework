import os

# Prevent OpenMP conflict between PyTorch and FAISS on macOS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from src.config.constants import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_MODEL,
    DEFAULT_TOP_K_PATENTS,
    DEFAULT_TOP_K_PAPERS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_WEIGHT_SEMANTIC,
    DEFAULT_WEIGHT_OVERLAP,
    DEFAULT_WEIGHT_CLAIM,
    DEFAULT_FEATURE_MATCH_THRESHOLD,
    DEFAULT_FEATURE_PARTIAL_THRESHOLD,
    DEFAULT_FEATURE_UNCERTAIN_THRESHOLD,
    PROJECT_ROOT,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # LLM Settings
    llm_provider: str = Field(default=DEFAULT_LLM_PROVIDER, validation_alias="LLM_PROVIDER")
    llm_model: str = Field(default=DEFAULT_LLM_MODEL, validation_alias="LLM_MODEL")
    openai_api_key: Optional[str] = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(default=None, validation_alias="OPENAI_BASE_URL")
    temperature: float = Field(default=0.2, validation_alias="TEMPERATURE")
    max_tokens: int = Field(default=2048, validation_alias="MAX_TOKENS")

    # Embedding & Hardware
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL, validation_alias="EMBEDDING_MODEL")
    device: str = Field(default="cpu", validation_alias="DEVICE")

    # Vector Store
    vector_store: str = Field(default="faiss", validation_alias="VECTOR_STORE")

    # Retrieval
    top_k_patents: int = Field(default=DEFAULT_TOP_K_PATENTS, validation_alias="TOP_K_PATENTS")
    top_k_papers: int = Field(default=DEFAULT_TOP_K_PAPERS, validation_alias="TOP_K_PAPERS")
    similarity_threshold: float = Field(default=0.20, validation_alias="SIMILARITY_THRESHOLD")

    # Chunking
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=DEFAULT_CHUNK_OVERLAP, validation_alias="CHUNK_OVERLAP")

    # Novelty Weights (Prototype Heuristic Parameters - Not Empirically Validated)
    weight_semantic: float = Field(default=DEFAULT_WEIGHT_SEMANTIC, validation_alias="NOVELTY_WEIGHT_SEMANTIC")
    weight_overlap: float = Field(default=DEFAULT_WEIGHT_OVERLAP, validation_alias="NOVELTY_WEIGHT_TECHNICAL")
    weight_claim: float = Field(default=DEFAULT_WEIGHT_CLAIM, validation_alias="NOVELTY_WEIGHT_CLAIM")

    # Semantic Feature Comparison Thresholds (Prototype Parameters)
    feature_match_threshold: float = Field(default=DEFAULT_FEATURE_MATCH_THRESHOLD, validation_alias="FEATURE_MATCH_THRESHOLD")
    feature_partial_threshold: float = Field(default=DEFAULT_FEATURE_PARTIAL_THRESHOLD, validation_alias="FEATURE_PARTIAL_THRESHOLD")
    feature_uncertain_threshold: float = Field(default=DEFAULT_FEATURE_UNCERTAIN_THRESHOLD, validation_alias="FEATURE_UNCERTAIN_THRESHOLD")


# Global singleton
settings = Settings()

