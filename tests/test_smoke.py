"""
Smoke tests for Prime Lands RAG Platform.
Tests core modules without requiring live API calls or Qdrant.
Run with: pytest tests/ -v
"""

import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock


# ─── Fixtures ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent


@pytest.fixture
def corpus_path():
    return ROOT / "data" / "primelands_corpus.jsonl"


@pytest.fixture
def outputs_dir():
    return ROOT / "outputs"


@pytest.fixture
def chunks_dir():
    return ROOT / "data" / "chunks"


# ─── Part 1: Data Pipeline ────────────────────────────────────────────────────

class TestCorpus:
    def test_corpus_exists(self, corpus_path):
        """Corpus file must exist after crawling."""
        assert corpus_path.exists(), f"Corpus not found at {corpus_path}"

    def test_corpus_not_empty(self, corpus_path):
        """Corpus must have at least 10 properties."""
        lines = [l for l in corpus_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) >= 10, f"Expected >=10 properties, got {len(lines)}"

    def test_corpus_valid_jsonl(self, corpus_path):
        """Every line in corpus must be valid JSON."""
        for i, line in enumerate(corpus_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON on line {i+1}: {e}")
            assert "url" in obj or "title" in obj, f"Line {i+1} missing url/title"

    def test_corpus_has_required_fields(self, corpus_path):
        """Each property must have title and description."""
        for i, line in enumerate(corpus_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            obj = json.loads(line)
            assert "title" in obj, f"Line {i+1} missing 'title'"
            assert "description" in obj, f"Line {i+1} missing 'description'"


# ─── Part 2: Chunking ─────────────────────────────────────────────────────────

class TestChunking:
    def test_semantic_chunks_exist(self, chunks_dir):
        """Semantic chunks file must exist."""
        chunk_file = chunks_dir / "semantic_chunks.jsonl"
        assert chunk_file.exists(), f"Semantic chunks not found at {chunk_file}"

    def test_semantic_chunks_not_empty(self, chunks_dir):
        """Must have at least 5 semantic chunks."""
        chunk_file = chunks_dir / "semantic_chunks.jsonl"
        lines = [l for l in chunk_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) >= 5, f"Expected >=5 chunks, got {len(lines)}"

    def test_chunking_comparison_exists(self, outputs_dir):
        """chunking_comparison.csv must exist."""
        assert (outputs_dir / "chunking_comparison.csv").exists()

    def test_chunking_comparison_has_all_strategies(self, outputs_dir):
        """Must have all 5 chunking strategies."""
        import csv
        with open(outputs_dir / "chunking_comparison.csv") as f:
            reader = csv.DictReader(f)
            strategies = {row["Strategy"] for row in reader}
        expected = {"semantic", "fixed", "sliding", "parent_child", "late"}
        assert expected == strategies, f"Missing strategies: {expected - strategies}"


# ─── Part 3: Intelligence Layers ─────────────────────────────────────────────

class TestServiceModels:
    def test_rag_service_importable(self):
        """RAGService must be importable."""
        from prime_lands.services.rag_service import RAGService
        assert RAGService is not None

    def test_cag_service_importable(self):
        """CAGService must be importable."""
        from prime_lands.services.cag_service import CAGService
        assert CAGService is not None

    def test_crag_service_importable(self):
        """CRAGService must be importable."""
        from prime_lands.services.crag_service import CRAGService
        assert CRAGService is not None

    def test_cag_stats_exists(self, outputs_dir):
        """cag_stats.json must exist."""
        assert (outputs_dir / "cag_stats.json").exists()

    def test_cag_stats_valid(self, outputs_dir):
        """cag_stats.json must have required fields."""
        with open(outputs_dir / "cag_stats.json") as f:
            stats = json.load(f)
        required = {"service", "total_queries", "cache_hits", "cache_misses", "hit_rate_pct"}
        missing = required - set(stats.keys())
        assert not missing, f"cag_stats.json missing fields: {missing}"

    def test_crag_impact_exists(self, outputs_dir):
        """crag_impact.csv must exist."""
        assert (outputs_dir / "crag_impact.csv").exists()

    def test_crag_impact_has_20_queries(self, outputs_dir):
        """crag_impact.csv must have 20 query rows."""
        import csv
        with open(outputs_dir / "crag_impact.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 20, f"Expected 20 rows, got {len(rows)}"

    def test_crag_impact_has_all_categories(self, outputs_dir):
        """crag_impact.csv must cover all 4 query categories."""
        import csv
        with open(outputs_dir / "crag_impact.csv") as f:
            rows = list(csv.DictReader(f))
        categories = {r["category"] for r in rows}
        assert categories == {"clear", "vague", "complex", "edge"}


# ─── Part 4: Performance Arena ────────────────────────────────────────────────

class TestPerformanceResults:
    def test_performance_comparison_exists(self, outputs_dir):
        """performance_comparison.csv must exist."""
        assert (outputs_dir / "performance_comparison.csv").exists()

    def test_performance_comparison_has_both_services(self, outputs_dir):
        """Must have RAG and CRAG rows."""
        import csv
        with open(outputs_dir / "performance_comparison.csv") as f:
            rows = list(csv.DictReader(f))
        services = {r["Service"] for r in rows}
        assert "RAG" in services and "CRAG" in services

    def test_performance_scores_are_valid(self, outputs_dir):
        """All metric scores must be floats between 0 and 1."""
        import csv
        with open(outputs_dir / "performance_comparison.csv") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for metric in ["Faithfulness", "Context Recall", "Context Precision"]:
                val = float(row[metric])
                assert 0.0 <= val <= 1.0, f"{row['Service']} {metric}={val} out of range [0,1]"

    def test_faithfulness_above_threshold(self, outputs_dir):
        """Faithfulness must be >= 0.8 for both services (production quality bar)."""
        import csv
        with open(outputs_dir / "performance_comparison.csv") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            faith = float(row["Faithfulness"])
            assert faith >= 0.8, f"{row['Service']} Faithfulness={faith} below 0.8 threshold"

    def test_cost_analysis_exists(self, outputs_dir):
        """cost_analysis.json must exist."""
        assert (outputs_dir / "cost_analysis.json").exists()

    def test_cost_analysis_has_three_scales(self, outputs_dir):
        """cost_analysis.json must have 3 scale projections."""
        with open(outputs_dir / "cost_analysis.json") as f:
            data = json.load(f)
        assert len(data["scales"]) == 3, f"Expected 3 scales, got {len(data['scales'])}"


# ─── Part 5: Engineering Report ───────────────────────────────────────────────

class TestReport:
    def test_engineering_report_md_exists(self):
        """ENGINEERING_REPORT.md must exist."""
        assert (ROOT / "ENGINEERING_REPORT.md").exists()

    def test_engineering_report_pdf_exists(self):
        """report/engineering_report.pdf must exist."""
        assert (ROOT / "report" / "engineering_report.pdf").exists()

    def test_engineering_report_pdf_not_empty(self):
        """PDF must be at least 5KB."""
        pdf_path = ROOT / "report" / "engineering_report.pdf"
        size = pdf_path.stat().st_size
        assert size >= 5000, f"PDF too small: {size} bytes"

    def test_engineering_report_has_sections(self):
        """ENGINEERING_REPORT.md must have all 7 required sections."""
        content = (ROOT / "ENGINEERING_REPORT.md").read_text(encoding="utf-8")
        required_sections = [
            "Executive Summary",
            "System Architecture",
            "Performance Arena",
            "Chunking",
            "Technical Constraints",
            "Cost Projection",
            "Future Work",
        ]
        for section in required_sections:
            assert section in content, f"Missing section: '{section}'"


# ─── Config & Package ─────────────────────────────────────────────────────────

class TestConfig:
    def test_config_yaml_exists(self):
        """config.yaml must exist."""
        assert (ROOT / "config.yaml").exists()

    def test_env_example_exists(self):
        """.env.example must exist."""
        assert (ROOT / ".env.example").exists()

    def test_package_importable(self):
        """prime_lands package must be importable."""
        import prime_lands
        assert prime_lands is not None

    def test_config_loadable(self):
        """Config must load without errors."""
        from prime_lands.config import load_config
        cfg = load_config(ROOT / "config.yaml")
        assert cfg is not None
        assert cfg.llm is not None
        assert cfg.retrieval is not None
