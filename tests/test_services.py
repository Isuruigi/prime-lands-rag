"""
test_services.py — Unit tests for RAG / CAG / CRAG services.
Tests service initialization, config loading, cache logic, CRAG grading.
"""
import json
import re
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

ROOT = Path(__file__).parent.parent


# ── Config ───────────────────────────────────────────────────────────────
class TestConfig:
    def test_load_config_returns_platform_config(self):
        from prime_lands.config import load_config, PlatformConfig
        cfg = load_config(ROOT / "config.yaml")
        assert isinstance(cfg, PlatformConfig)

    def test_embedding_provider_set(self):
        from prime_lands.config import load_config
        cfg = load_config(ROOT / "config.yaml")
        assert cfg.embeddings.provider in ("openai", "fastembed")

    def test_llm_model_set(self):
        from prime_lands.config import load_config
        cfg = load_config(ROOT / "config.yaml")
        assert cfg.llm.model != ""

    def test_qdrant_collections_count(self):
        from prime_lands.config import load_config
        cfg = load_config(ROOT / "config.yaml")
        cols = cfg.qdrant.collections
        assert len(cols) == 5, f"Expected 5 collections, found {len(cols)}"

    def test_crag_thresholds_valid(self):
        from prime_lands.config import load_config
        cfg = load_config(ROOT / "config.yaml")
        assert 0.0 < cfg.crag.doc_relevance_threshold < 1.0
        assert 0.0 < cfg.crag.global_confidence_threshold < 1.0
        assert cfg.crag.ambiguous_band > 0.0

    def test_retrieval_reranker_model_set(self):
        from prime_lands.config import load_config
        cfg = load_config(ROOT / "config.yaml")
        assert "cross-encoder" in cfg.retrieval.reranker_model.lower()


# ── Exceptions ───────────────────────────────────────────────────────────
class TestExceptions:
    def test_custom_exceptions_importable(self):
        from prime_lands.exceptions import (
            PrimeLandsError, CrawlerError, ChunkingError,
            IndexingError, ServiceError,
        )

    def test_exception_hierarchy(self):
        from prime_lands.exceptions import PrimeLandsError, ServiceError
        assert issubclass(ServiceError, PrimeLandsError)

    def test_collection_exists_error(self):
        from prime_lands.exceptions import CollectionExistsError, IndexingError
        assert issubclass(CollectionExistsError, IndexingError)


# ── RAGService ───────────────────────────────────────────────────────────
class TestRAGService:
    def test_importable(self):
        from prime_lands.services.rag_service import RAGService

    def test_rag_result_model(self):
        from prime_lands.services.base import QueryResult
        r = QueryResult(
            answer="test",
            sources=[],
            query="q",
            latency_ms=100.0,
            cost=0.001,
            metadata={},
        )
        assert r.answer == "test"
        assert r.latency_ms == 100.0


# ── CAGService ───────────────────────────────────────────────────────────
class TestCAGService:
    def test_importable(self):
        from prime_lands.services.cag_service import CAGService

    def test_stats_structure_from_file(self):
        stats_path = ROOT / "outputs" / "cag_stats.json"
        assert stats_path.exists()
        with open(stats_path) as f:
            stats = json.load(f)
        for key in ("total_queries", "cache_hits", "cache_misses", "hit_rate_pct"):
            assert key in stats, f"Missing key: {key}"

    def test_hit_rate_valid(self):
        stats_path = ROOT / "outputs" / "cag_stats.json"
        with open(stats_path) as f:
            stats = json.load(f)
        assert 0.0 <= stats["hit_rate_pct"] <= 100.0

    def test_queries_and_hits_consistent(self):
        stats_path = ROOT / "outputs" / "cag_stats.json"
        with open(stats_path) as f:
            stats = json.load(f)
        assert stats["cache_hits"] + stats["cache_misses"] == stats["total_queries"]


# ── CRAGService ──────────────────────────────────────────────────────────
class TestCRAGService:
    def test_importable(self):
        from prime_lands.services.crag_service import CRAGService

    def test_grade_parser_regex(self):
        """Verify the regex in _grade_document extracts floats correctly."""
        pattern = r'\b(1\.0|0\.\d+|\d+\.\d+|[01])\b'
        test_cases = [
            ("0.7", 0.7),
            ("0.7\n\nThis is relevant.", 0.7),
            ("Score: 0.85 out of 1.0", 0.85),
            ("  1.0  ", 1.0),
            ("0", 0.0),
        ]
        for raw, expected in test_cases:
            match = re.search(pattern, raw.strip())
            assert match, f"No match for '{raw}'"
            assert abs(float(match.group(1)) - expected) < 0.01

    def test_crag_impact_exists(self):
        crag_path = ROOT / "outputs" / "crag_impact.csv"
        assert crag_path.exists(), "crag_impact.csv missing"

    def test_crag_impact_has_rows(self):
        import pandas as pd
        crag_path = ROOT / "outputs" / "crag_impact.csv"
        df = pd.read_csv(crag_path)
        assert len(df) >= 5, f"Expected 5+ rows, found {len(df)}"


# ── Qdrant Indexer (mock) ────────────────────────────────────────────────
class TestQdrantIndexer:
    def test_importable(self):
        from prime_lands.indexing.qdrant_indexer import QdrantIndexer

    def test_has_hybrid_search(self):
        from prime_lands.indexing.qdrant_indexer import QdrantIndexer
        assert hasattr(QdrantIndexer, "hybrid_search")

    def test_has_reranker(self):
        from prime_lands.indexing.qdrant_indexer import QdrantIndexer
        assert hasattr(QdrantIndexer, "_get_reranker")

    def test_has_index_chunks(self):
        from prime_lands.indexing.qdrant_indexer import QdrantIndexer
        assert hasattr(QdrantIndexer, "index_chunks")

    def test_has_create_collection(self):
        from prime_lands.indexing.qdrant_indexer import QdrantIndexer
        assert hasattr(QdrantIndexer, "create_collection")
