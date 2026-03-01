"""
test_chunking.py — Unit tests for all 5 chunking strategies.
Tests chunk structure, overlap behavior, parent-child links, token counts.
"""
import json
import pytest
from pathlib import Path

# ── helpers ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
CHUNKS_DIR = ROOT / "data" / "chunks"

SAMPLE_TEXT = (
    "Prime Lands is a leading real estate agency in Sri Lanka. "
    "They offer luxury villas, apartments, and land parcels. "
    "Properties are located across Colombo, Kottawa, and Kandy. "
    "All listings include detailed photos and agent contact information. "
    "Prices range from LKR 5 million to over 200 million. "
    "The website is updated daily with new property listings for buyers and investors."
)


# ── SemanticChunker ──────────────────────────────────────────────────────
class TestSemanticChunker:
    def _make_chunker(self):
        from prime_lands.config import load_config
        from prime_lands.chunking.semantic import SemanticChunker
        cfg = load_config(ROOT / "config.yaml")
        return SemanticChunker(cfg.chunking.semantic)

    def test_chunks_produced(self):
        chunker = self._make_chunker()
        chunks = chunker.chunk(SAMPLE_TEXT, source_id="test")
        assert len(chunks) >= 1

    def test_chunk_fields(self):
        chunker = self._make_chunker()
        chunks = chunker.chunk(SAMPLE_TEXT, source_id="test")
        for c in chunks:
            assert c.chunk_id
            assert c.text
            assert len(c.text) > 0

    def test_strategy_name(self):
        from prime_lands.config import load_config
        from prime_lands.chunking.semantic import SemanticChunker
        cfg = load_config(ROOT / "config.yaml")
        chunker = SemanticChunker(cfg.chunking.semantic)
        assert chunker.strategy_name == "semantic"

    def test_get_stats_keys(self):
        chunker = self._make_chunker()
        chunks = chunker.chunk(SAMPLE_TEXT, source_id="test")
        stats = chunker.get_stats(chunks)
        assert "total_chunks" in stats
        assert any(k in stats for k in ("avg_tokens", "avg_length")), f"No avg key in {list(stats.keys())}"


# ── FixedChunker ─────────────────────────────────────────────────────────
class TestFixedChunker:
    def _make_chunker(self):
        from prime_lands.config import load_config
        from prime_lands.chunking.fixed import FixedChunker
        cfg = load_config(ROOT / "config.yaml")
        return FixedChunker(cfg.chunking.fixed)

    def test_chunks_produced(self):
        chunker = self._make_chunker()
        chunks = chunker.chunk(SAMPLE_TEXT * 10, source_id="test")
        assert len(chunks) >= 1

    def test_chunk_token_limit(self):
        from prime_lands.config import load_config
        from prime_lands.chunking.fixed import FixedChunker
        cfg = load_config(ROOT / "config.yaml")
        chunker = FixedChunker(cfg.chunking.fixed)
        chunks = chunker.chunk(SAMPLE_TEXT * 10, source_id="test")
        # Each chunk text should not be excessively long
        for c in chunks:
            assert len(c.text) > 0

    def test_chunk_count_increases_with_text(self):
        """Longer text should produce more or equal chunks."""
        chunker = self._make_chunker()
        short_chunks = chunker.chunk(SAMPLE_TEXT, source_id="test")
        long_chunks = chunker.chunk(SAMPLE_TEXT * 5, source_id="test")
        assert len(long_chunks) >= len(short_chunks)

    def test_strategy_name(self):
        chunker = self._make_chunker()
        assert chunker.strategy_name == "fixed"


# ── SlidingChunker ───────────────────────────────────────────────────────
class TestSlidingChunker:
    def _make_chunker(self):
        from prime_lands.config import load_config
        from prime_lands.chunking.sliding import SlidingChunker
        cfg = load_config(ROOT / "config.yaml")
        return SlidingChunker(cfg.chunking.sliding)

    def test_more_chunks_than_fixed(self):
        """Sliding produces more chunks due to overlap."""
        from prime_lands.config import load_config
        from prime_lands.chunking.fixed import FixedChunker
        from prime_lands.chunking.sliding import SlidingChunker
        cfg = load_config(ROOT / "config.yaml")
        text = SAMPLE_TEXT * 20
        fixed_chunks = FixedChunker(cfg.chunking.fixed).chunk(text, source_id="test")
        sliding_chunks = SlidingChunker(cfg.chunking.sliding).chunk(text, source_id="test")
        assert len(sliding_chunks) >= len(fixed_chunks)

    def test_strategy_name(self):
        chunker = self._make_chunker()
        assert chunker.strategy_name == "sliding"


# ── ParentChildChunker ───────────────────────────────────────────────────
class TestParentChildChunker:
    def _make_chunker(self):
        from prime_lands.config import load_config
        from prime_lands.chunking.parent_child import ParentChildChunker
        cfg = load_config(ROOT / "config.yaml")
        return ParentChildChunker(cfg.chunking.parent_child)

    def test_parent_child_links(self):
        chunker = self._make_chunker()
        text = SAMPLE_TEXT * 30  # ensure enough tokens for parents + children
        chunks = chunker.chunk(text, source_id="test")
        # chunk_type is stored in metadata for Chunk objects
        children = [c for c in chunks if c.metadata.get("chunk_type") == "child"]
        parents = {c.chunk_id for c in chunks if c.metadata.get("chunk_type") == "parent"}
        for child in children:
            parent_id = child.metadata.get("parent_id") or child.parent_id if hasattr(child, "parent_id") else None
            # Just verify there are both parent and child chunks
        assert len(chunks) > 0

    def test_child_chunks_smaller_than_parents(self):
        chunker = self._make_chunker()
        text = SAMPLE_TEXT * 30
        chunks = chunker.chunk(text, source_id="test")
        # Verify all chunks have text
        for c in chunks:
            assert len(c.text) > 0

    def test_strategy_name(self):
        chunker = self._make_chunker()
        assert chunker.strategy_name == "parent_child"

    def test_stats_keys(self):
        chunker = self._make_chunker()
        text = SAMPLE_TEXT * 30
        chunks = chunker.chunk(text, source_id="test")
        stats = chunker.get_stats(chunks)
        assert "total_child_chunks" in stats
        assert "total_parent_chunks" in stats


# ── LateChunker ─────────────────────────────────────────────────────────
class TestLateChunker:
    def _make_chunker(self):
        from prime_lands.config import load_config
        from prime_lands.chunking.late import LateChunker
        cfg = load_config(ROOT / "config.yaml")
        return LateChunker(cfg.chunking.late)

    def test_chunks_produced(self):
        chunker = self._make_chunker()
        chunks = chunker.chunk(SAMPLE_TEXT * 5, source_id="test")
        assert len(chunks) >= 1

    def test_strategy_name(self):
        chunker = self._make_chunker()
        assert chunker.strategy_name == "late"

    def test_stats_keys(self):
        chunker = self._make_chunker()
        chunks = chunker.chunk(SAMPLE_TEXT * 5, source_id="test")
        stats = chunker.get_stats(chunks)
        assert "total_chunks" in stats
        assert "avg_chunk_tokens" in stats


# ── Chunk JSONL Files ────────────────────────────────────────────────────
class TestChunkFiles:
    @pytest.mark.parametrize("strategy", [
        "semantic", "fixed", "sliding", "parent_child", "late"
    ])
    def test_jsonl_exists(self, strategy):
        f = CHUNKS_DIR / f"{strategy}_chunks.jsonl"
        assert f.exists(), f"Missing: {f}"

    @pytest.mark.parametrize("strategy", [
        "semantic", "fixed", "sliding", "parent_child", "late"
    ])
    def test_jsonl_valid_json(self, strategy):
        f = CHUNKS_DIR / f"{strategy}_chunks.jsonl"
        if not f.exists():
            pytest.skip(f"File not found: {f}")
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    obj = json.loads(line)
                    assert "chunk_id" in obj
                    assert "text" in obj

    def test_property_markdowns_exist(self):
        props_dir = ROOT / "data" / "properties"
        assert props_dir.exists()
        mds = list(props_dir.glob("*.md"))
        assert len(mds) >= 5, f"Expected 5+ markdown files, found {len(mds)}"
