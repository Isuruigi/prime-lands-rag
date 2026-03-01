"""
Chunking Lab — full pipeline per the submission checklist spec.

Part 2: Saves all 5 chunk JSONL files + property markdown files.
Part 4A: RAGAS evaluation (faithfulness, context_recall, context_precision)
         + latency (avg_latency_ms, p95_latency_ms) for each strategy.

Outputs:
  data/chunks/{strategy}_chunks.jsonl        (Part 2)
  data/properties/*.md                        (Part 2)
  outputs/chunking_comparison.csv             (Part 4A — RAGAS metrics)
"""

import sys
import json
import time
import asyncio
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, List, Optional
from dotenv import load_dotenv

# ── RAGAS / LangChain imports (same pattern as run_performance.py) ────────────
import anthropic as anthropic_sdk
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_community.embeddings import FastEmbedEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, context_recall, context_precision
from datasets import Dataset

# ── Prime Lands imports ───────────────────────────────────────────────────────
from prime_lands.config import load_config
from prime_lands.chunking.semantic import SemanticChunker
from prime_lands.chunking.fixed import FixedChunker
from prime_lands.chunking.sliding import SlidingChunker
from prime_lands.chunking.parent_child import ParentChildChunker
from prime_lands.chunking.late import LateChunker
from prime_lands.crawler.models import PropertyDocument
from prime_lands.indexing.qdrant_indexer import QdrantIndexer
from prime_lands.services.rag_service import RAGService
from prime_lands.logger import setup_logger


# ── Anthropic LangChain wrapper (matches run_performance.py) ─────────────────
class AnthropicLLM(LLM):
    """Minimal LangChain wrapper around the Anthropic SDK for RAGAS evaluation."""
    model_name: str = "claude-3-haiku-20240307"
    max_tokens: int = 512
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "anthropic"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        client = anthropic_sdk.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


# ── Test queries with ground truth for RAGAS ─────────────────────────────────
TEST_CASES = [
    {
        "question": "What 3 bedroom properties are available?",
        "ground_truth": "Multiple 3-bedroom properties are available including houses, apartments, and villas with various amenities.",
    },
    {
        "question": "Are there luxury villas for sale?",
        "ground_truth": "Luxury villas are available with premium amenities such as swimming pools, gyms, and security systems.",
    },
    {
        "question": "What properties are available near Colombo?",
        "ground_truth": "Various residential properties including apartments and houses are available in and around Colombo.",
    },
    {
        "question": "Do any properties have swimming pools?",
        "ground_truth": "Several properties include swimming pools as an amenity, particularly luxury developments.",
    },
    {
        "question": "What is the price range for 2 bedroom apartments?",
        "ground_truth": "2-bedroom apartment prices vary by location and amenities, with options at various price points.",
    },
    {
        "question": "Are there properties with ocean views?",
        "ground_truth": "Some coastal properties and high-rise apartments offer ocean or sea views.",
    },
    {
        "question": "What properties are available in Negombo?",
        "ground_truth": "Properties in Negombo include residential apartments and villas near the beach.",
    },
    {
        "question": "Which properties offer parking facilities?",
        "ground_truth": "Many modern developments include dedicated parking facilities as part of their amenity package.",
    },
    {
        "question": "Are there properties suitable for families?",
        "ground_truth": "Family-friendly properties with multiple bedrooms, gardens, and community facilities are available.",
    },
    {
        "question": "What commercial properties are available?",
        "ground_truth": "Commercial property listings include offices, shops, and mixed-use developments.",
    },
]


async def evaluate_strategy_with_ragas(
    rag_service: RAGService,
    collection_name: str,
    eval_llm: AnthropicLLM,
    eval_embeddings: FastEmbedEmbeddings,
    strategy_name: str,
) -> dict:
    """
    Run 10 test queries against a Qdrant collection and evaluate with RAGAS.

    Returns dict with: faithfulness, context_recall, context_precision,
    avg_latency_ms, p95_latency_ms, retrieved_count.
    """
    print(f"  [{strategy_name}] Querying {len(TEST_CASES)} test cases...")
    results = []
    latencies = []

    for case in TEST_CASES:
        t0 = time.time()
        res = await rag_service.query(case["question"], collection_name=collection_name)
        elapsed_ms = (time.time() - t0) * 1000
        latencies.append(elapsed_ms)
        results.append({
            "question":     case["question"],
            "answer":       res.answer,
            "contexts":     res.contexts if res.contexts else [""],
            "ground_truth": case["ground_truth"],
        })

    dataset = Dataset.from_dict({
        "question":     [r["question"]     for r in results],
        "answer":       [r["answer"]       for r in results],
        "contexts":     [r["contexts"]     for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })

    print(f"  [{strategy_name}] Running RAGAS evaluation...")
    def _to_score(val) -> float:
        """Safely convert RAGAS metric value to float (handles list or scalar)."""
        if isinstance(val, (list, tuple)):
            valid = [v for v in val if v is not None and not np.isnan(float(v))]
            return float(np.mean(valid)) if valid else 0.0
        try:
            v = float(val)
            return 0.0 if np.isnan(v) else v
        except Exception:
            return 0.0

    try:
        scores = evaluate(
            dataset,
            metrics=[faithfulness, context_recall, context_precision],
            llm=eval_llm,
            embeddings=eval_embeddings,
        )
        faith     = round(_to_score(scores["faithfulness"]),       3)
        recall    = round(_to_score(scores["context_recall"]),     3)
        precision = round(_to_score(scores["context_precision"]),  3)
    except Exception as e:
        print(f"  [{strategy_name}] RAGAS failed: {e}. Using fallback zeros.")
        faith = recall = precision = 0.0

    avg_lat = round(float(np.mean(latencies)),                    1)
    p95_lat = round(float(np.percentile(latencies, 95)),          1)
    avg_retrieved = round(
        sum(len(r["contexts"]) for r in results) / len(results), 1
    )

    return {
        "strategy":        strategy_name,
        "faithfulness":    faith,
        "context_recall":  recall,
        "context_precision": precision,
        "avg_latency_ms":  avg_lat,
        "p95_latency_ms":  p95_lat,
        "retrieved_count": avg_retrieved,
    }


async def main():
    setup_logger(level="INFO")

    root_dir    = Path.cwd()
    data_dir    = root_dir / "data"
    outputs_dir = root_dir / "outputs"
    chunks_dir  = data_dir / "chunks"
    outputs_dir.mkdir(exist_ok=True)
    chunks_dir.mkdir(exist_ok=True)

    load_dotenv(root_dir / ".env")
    cfg = load_config(root_dir / "config.yaml")

    # ── Load corpus ──────────────────────────────────────────────────────────
    corpus_file = data_dir / "primelands_corpus.jsonl"
    if not corpus_file.exists():
        print(f"Error: {corpus_file} not found. Run the crawler first.")
        return

    properties = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            properties.append(PropertyDocument(**json.loads(line)))
    print(f"Loaded {len(properties)} properties")

    # ── Chunkers ─────────────────────────────────────────────────────────────
    chunkers = {
        "semantic":     SemanticChunker(cfg.chunking.semantic),
        "fixed":        FixedChunker(cfg.chunking.fixed),
        "sliding":      SlidingChunker(cfg.chunking.sliding),
        "parent_child": ParentChildChunker(cfg.chunking.parent_child),
        "late":         LateChunker(cfg.chunking.late),
    }

    # ── Part 2: Save chunk JSONL files (all properties) ──────────────────────
    print("\n[Part 2] Saving 5 strategy chunk files...")
    all_chunks: dict = {}
    for strategy_name, chunker in chunkers.items():
        full_chunks = []
        for prop in properties:
            full_chunks.extend(chunker.chunk(prop.to_rag_text(), source_id=prop.property_id))
        all_chunks[strategy_name] = full_chunks

        out_file = chunks_dir / f"{strategy_name}_chunks.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for chunk in full_chunks:
                f.write(chunk.model_dump_json() + "\n")
        print(f"  Saved {len(full_chunks):3d} {strategy_name} chunks -> {out_file.name}")

    # ── Part 2: Save property markdown files ─────────────────────────────────
    properties_dir = data_dir / "properties"
    properties_dir.mkdir(exist_ok=True)
    print(f"\n[Part 2] Generating property markdown files -> data/properties/")
    for prop in properties:
        md_lines = [
            f"# {prop.title}", "",
            f"**URL:** {prop.url}",
            f"**Type:** {prop.property_type or 'N/A'}",
            f"**Price:** {prop.price or 'N/A'}",
            f"**Address:** {prop.address or 'N/A'}",
            f"**Bedrooms:** {prop.bedrooms or 'N/A'}",
            f"**Bathrooms:** {prop.bathrooms or 'N/A'}",
            f"**Sqft:** {prop.sqft or 'N/A'}",
            f"**Agent:** {prop.agent or 'N/A'}",
            f"**Crawled At:** {prop.crawled_at}", "",
            "## Description", "",
            prop.description or "No description available.", "",
        ]
        if prop.amenities:
            md_lines += ["## Amenities", ""]
            for a in prop.amenities:
                md_lines.append(f"- {a}")
            md_lines.append("")

        safe_id = prop.property_id.replace("/", "_").replace(":", "_")[:80]
        (properties_dir / f"{safe_id}.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  Saved {len(properties)} property markdown files")

    # ── Part 4A: RAGAS evaluation per strategy ────────────────────────────────
    print("\n[Part 4A] RAGAS evaluation per chunking strategy...")
    print("  Initialising indexer and RAG service...")

    indexer     = QdrantIndexer(cfg)
    rag_service = RAGService(cfg, indexer)
    eval_llm    = AnthropicLLM(model_name="claude-3-haiku-20240307")
    eval_embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # Map strategy -> Qdrant collection name
    collection_map = {
        "semantic":     cfg.qdrant.collections.get("semantic",     "primelands_semantic"),
        "fixed":        cfg.qdrant.collections.get("fixed",        "primelands_fixed"),
        "sliding":      cfg.qdrant.collections.get("sliding",      "primelands_sliding"),
        "parent_child": cfg.qdrant.collections.get("parent_child", "primelands_parent_child"),
        "late":         cfg.qdrant.collections.get("late",         "primelands_late"),
    }

    ragas_rows = []
    for strategy_name, collection_name in collection_map.items():
        print(f"\n  Strategy: {strategy_name} (collection: {collection_name})")
        try:
            row = await evaluate_strategy_with_ragas(
                rag_service, collection_name, eval_llm, eval_embeddings, strategy_name
            )
        except Exception as e:
            print(f"  ERROR evaluating {strategy_name}: {e}")
            row = {
                "strategy": strategy_name,
                "faithfulness": 0.0,
                "context_recall": 0.0,
                "context_precision": 0.0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "retrieved_count": 0.0,
            }
        ragas_rows.append(row)
        print(f"  -> faithfulness={row['faithfulness']}, "
              f"context_recall={row['context_recall']}, "
              f"context_precision={row['context_precision']}, "
              f"avg_latency_ms={row['avg_latency_ms']}")

    # ── Also add structural stats ────────────────────────────────────────────
    struct_notes = {
        "semantic":     "Selected for production — best context coherence",
        "fixed":        "Breaks topic boundaries; not recommended",
        "sliding":      "3x redundancy; high storage cost",
        "parent_child": "Hierarchical parent-child linkage",
        "late":         "Context-enriched; late chunking (JINA-style)",
    }
    for row in ragas_rows:
        name   = row["strategy"]
        chunks = all_chunks[name]
        s      = chunkers[name].get_stats(chunks)
        total  = s.get("total_chunks", s.get("total_child_chunks", len(chunks)))
        avg_ch = round(sum(len(c.text) for c in chunks) / len(chunks), 1) if chunks else 0.0
        row["total_chunks"] = total
        row["avg_chars"]    = avg_ch
        row["notes"]        = struct_notes.get(name, "")

    # Reorder columns for clarity
    cols = [
        "strategy", "faithfulness", "context_recall", "context_precision",
        "avg_latency_ms", "p95_latency_ms", "retrieved_count",
        "total_chunks", "avg_chars", "notes",
    ]
    df = pd.DataFrame(ragas_rows)[cols]
    out_csv = outputs_dir / "chunking_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[Done] Saved chunking_comparison.csv -> {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
