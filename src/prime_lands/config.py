"""
Pydantic Settings — validates all config at startup.
Fails fast on bad config (no mysterious runtime errors).
"""

from __future__ import annotations
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class CrawlerConfig(BaseModel):
    base_url: str
    max_pages: int = Field(gt=0)
    rate_limit_seconds: float = Field(ge=0.5)
    timeout_ms: int = Field(gt=0)
    headless: bool = True
    user_agent: str
    max_retries: int = Field(ge=0, le=10)
    backoff_base: float = Field(ge=1.0)
    backoff_max: float = Field(ge=1.0)
    respect_robots_txt: bool = True


class SemanticChunkConfig(BaseModel):
    model: str
    similarity_threshold: float = Field(ge=0.0, le=1.0)
    min_tokens: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    sentence_splitter: Literal["spacy", "nltk", "regex"] = "regex"


class FixedChunkConfig(BaseModel):
    chunk_size: int = Field(gt=0)
    overlap: int = Field(ge=0)

    @field_validator("overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        """Ensure overlap doesn't exceed chunk size."""
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("overlap must be less than chunk_size")
        return v


class SlidingChunkConfig(BaseModel):
    window_size: int = Field(gt=0)
    step_size: int = Field(gt=0)


class ParentChildConfig(BaseModel):
    parent_size: int = Field(gt=0)
    child_size: int = Field(gt=0)
    link_children: bool = True


class LateChunkConfig(BaseModel):
    context_window: int = Field(gt=0)
    prepend_ratio: float = Field(ge=0.0, le=0.5)


class ChunkingConfig(BaseModel):
    semantic: SemanticChunkConfig
    fixed: FixedChunkConfig
    sliding: SlidingChunkConfig
    parent_child: ParentChildConfig
    late: LateChunkConfig


class EmbeddingConfig(BaseModel):
    provider: Literal["openai", "huggingface", "fastembed"] = "openai"
    dense_model: str
    dense_dimensions: int = Field(gt=0)
    sparse_model: str
    batch_size: int = Field(gt=0)


class QdrantConfig(BaseModel):
    host: str = "localhost"
    port: int = 6333
    collections: dict[str, str]
    hybrid_fusion: Literal["rrf", "dbsf"] = "rrf"


class RetrievalConfig(BaseModel):
    top_k: int = Field(gt=0)
    rerank_top_n: int = Field(gt=0)
    reranker_model: str
    search_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    mmr_lambda: float = Field(ge=0.0, le=1.0)


class LLMConfig(BaseModel):
    provider: Literal["openai", "anthropic"] = "anthropic"
    model: str
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0)
    streaming: bool = False


class RAGConfig(BaseModel):
    system_prompt: str


class CAGConfig(BaseModel):
    similarity_threshold: float = Field(ge=0.0, le=1.0)
    cache_ttl_hours: int = Field(gt=0)
    faq_ttl_multiplier: int = Field(ge=1)
    max_history_size: int = Field(gt=0)


class CRAGConfig(BaseModel):
    doc_relevance_threshold: float = Field(ge=0.0, le=1.0)
    global_confidence_threshold: float = Field(ge=0.0, le=1.0)
    ambiguous_band: float = Field(ge=0.0, le=0.3)
    max_correction_iterations: int = Field(ge=1, le=5)
    rewrite_temperature: float = Field(ge=0.0, le=2.0)
    knowledge_refinement: bool = True


class EvaluationConfig(BaseModel):
    ragas_metrics: list[str]
    chunking_test_queries: int = Field(gt=0)
    cag_simulation_queries: int = Field(gt=0)
    crag_comparison_queries: int = Field(gt=0)
    repeat_query_ratio: float = Field(ge=0.0, le=1.0)


class PlatformConfig(BaseModel):
    """Root configuration model. Validates entire config.yaml at startup."""
    crawler: CrawlerConfig
    chunking: ChunkingConfig
    embeddings: EmbeddingConfig
    qdrant: QdrantConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    rag: RAGConfig
    cag: CAGConfig
    crag: CRAGConfig
    evaluation: EvaluationConfig


def load_config(path: str | Path = "config.yaml") -> PlatformConfig:
    """
    Load and validate platform configuration from YAML file.

    Args:
        path: Path to config.yaml

    Returns:
        Validated PlatformConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config values are invalid
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Pydantic validates ALL fields and raises detailed errors at import time
    return PlatformConfig(**raw)
