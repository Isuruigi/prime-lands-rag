"""
Generate crag_impact.csv (CRAG vs RAG per-query comparison)
and cost_analysis.json (3-scale cost projection).
"""

import asyncio
import json
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from prime_lands.config import load_config
from prime_lands.indexing.qdrant_indexer import QdrantIndexer
from prime_lands.services.rag_service import RAGService
from prime_lands.services.crag_service import CRAGService
from prime_lands.logger import setup_logger


async def main():
    setup_logger(level="INFO")

    root_dir = Path.cwd()
    outputs_dir = root_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    load_dotenv(root_dir / ".env")
    cfg = load_config(root_dir / "config.yaml")

    indexer = QdrantIndexer(cfg)
    rag_service = RAGService(cfg, indexer)
    crag_service = CRAGService(cfg, indexer)

    collection_name = "primelands_semantic"

    # 20 queries: 5 clear, 5 vague, 5 complex, 5 edge cases
    test_queries = [
        # Clear queries
        {"question": "What 3 bedroom properties are available?", "category": "clear"},
        {"question": "Show me apartments for sale in Colombo.", "category": "clear"},
        {"question": "What luxury villas are available?", "category": "clear"},
        {"question": "List properties with swimming pools.", "category": "clear"},
        {"question": "What are the prices of 2 bedroom apartments?", "category": "clear"},
        # Vague queries
        {"question": "Nice place near the city.", "category": "vague"},
        {"question": "Something affordable.", "category": "vague"},
        {"question": "Good investment property.", "category": "vague"},
        {"question": "Family home with space.", "category": "vague"},
        {"question": "Modern property.", "category": "vague"},
        # Complex queries
        {"question": "What are the typical amenities in luxury properties near Colombo city center?", "category": "complex"},
        {"question": "Compare 3 bedroom houses vs apartments in terms of price and amenities.", "category": "complex"},
        {"question": "Which properties offer ocean views and have more than 3 bedrooms?", "category": "complex"},
        {"question": "What is the price range for apartments with gym and parking facilities?", "category": "complex"},
        {"question": "Are there properties suitable for both residential and commercial use?", "category": "complex"},
        # Edge cases
        {"question": "Properties with helipads.", "category": "edge"},
        {"question": "Underwater properties.", "category": "edge"},
        {"question": "Properties available for barter trade.", "category": "edge"},
        {"question": "Do any properties have ocean views?", "category": "edge"},
        {"question": "Are there properties near Colombo city center?", "category": "edge"},
    ]

    print(f"\nRunning CRAG vs RAG comparison on {len(test_queries)} queries...")
    print("=" * 60)

    rows = []
    for i, case in enumerate(test_queries):
        question = case["question"]
        category = case["category"]
        print(f"\n[{i+1}/{len(test_queries)}] [{category.upper()}] {question[:60]}...")

        # RAG
        t0 = time.time()
        rag_res = await rag_service.query(question, collection_name=collection_name)
        rag_latency = (time.time() - t0) * 1000

        # CRAG
        t0 = time.time()
        crag_res = await crag_service.query(question, collection_name=collection_name)
        crag_latency = (time.time() - t0) * 1000

        corrections = crag_res.metadata.get("corrections_applied", 0)
        avg_grade = crag_res.metadata.get("avg_document_grade", 0.0)

        rows.append({
            "query_id": i + 1,
            "category": category,
            "question": question,
            "rag_latency_ms": round(rag_latency, 1),
            "crag_latency_ms": round(crag_latency, 1),
            "latency_overhead_ms": round(crag_latency - rag_latency, 1),
            "latency_overhead_pct": round((crag_latency - rag_latency) / rag_latency * 100, 1),
            "rag_cost": round(rag_res.cost, 6),
            "crag_cost": round(crag_res.cost, 6),
            "corrections_applied": corrections,
            "avg_document_grade": round(avg_grade, 3),
            "rag_contexts_count": len(rag_res.contexts),
            "crag_contexts_count": len(crag_res.contexts),
        })

        print(f"  RAG: {rag_latency:.0f}ms | CRAG: {crag_latency:.0f}ms | Corrections: {corrections} | Avg Grade: {avg_grade:.2f}")

    df = pd.DataFrame(rows)
    output_path = outputs_dir / "crag_impact.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved crag_impact.csv ({len(df)} rows)")

    # Summary stats
    print("\n--- Summary by Category ---")
    summary = df.groupby("category").agg(
        avg_rag_latency=("rag_latency_ms", "mean"),
        avg_crag_latency=("crag_latency_ms", "mean"),
        avg_overhead_pct=("latency_overhead_pct", "mean"),
        total_corrections=("corrections_applied", "sum"),
        avg_doc_grade=("avg_document_grade", "mean"),
    ).round(2)
    print(summary)

    # ─── Cost Analysis ────────────────────────────────────────────────
    print("\n\nGenerating cost_analysis.json...")

    # Actual measured costs from performance run
    rag_cost_per_query = 0.0073032 / 5   # from performance_comparison.csv
    crag_cost_per_query = 0.0070248 / 5

    # Embedding cost (local FastEmbed = $0)
    embed_cost_per_1k_tokens = 0.0  # local model

    # Qdrant storage: ~19 properties × ~5 chunks × ~200 tokens avg = ~19,000 tokens
    # At 384d vectors: ~19 chunks × 384 × 4 bytes ≈ 29KB → negligible
    qdrant_storage_gb = 0.001  # ~1MB
    qdrant_cost_per_gb_month = 0.10

    scales = [500, 2000, 10000]  # daily users
    days_per_month = 30
    avg_queries_per_user = 3  # avg queries per session

    cost_analysis = {
        "methodology": {
            "rag_cost_per_query_usd": round(rag_cost_per_query, 6),
            "crag_cost_per_query_usd": round(crag_cost_per_query, 6),
            "embedding_cost": "FREE (local FastEmbed BAAI/bge-small-en-v1.5)",
            "qdrant_storage_cost_usd_month": round(qdrant_storage_gb * qdrant_cost_per_gb_month, 4),
            "llm_model": "claude-3-haiku-20240307",
            "pricing_source": "Anthropic API (measured from actual runs)",
        },
        "scales": {}
    }

    for daily_users in scales:
        monthly_queries = daily_users * avg_queries_per_user * days_per_month
        rag_llm_cost = monthly_queries * rag_cost_per_query
        crag_llm_cost = monthly_queries * crag_cost_per_query
        storage_cost = qdrant_storage_gb * qdrant_cost_per_gb_month

        # CAG cache hit rate reduces LLM calls (assume 40% cache hit rate)
        cag_hit_rate = 0.40
        rag_with_cag_cost = rag_llm_cost * (1 - cag_hit_rate)
        cag_savings = rag_llm_cost - rag_with_cag_cost

        cost_analysis["scales"][f"{daily_users}_daily_users"] = {
            "daily_users": daily_users,
            "monthly_queries": monthly_queries,
            "rag_only": {
                "llm_cost_usd": round(rag_llm_cost, 2),
                "storage_cost_usd": round(storage_cost, 4),
                "total_usd": round(rag_llm_cost + storage_cost, 2),
                "cost_per_user_usd": round((rag_llm_cost + storage_cost) / daily_users / days_per_month, 4),
            },
            "rag_plus_cag": {
                "llm_cost_usd": round(rag_with_cag_cost, 2),
                "storage_cost_usd": round(storage_cost, 4),
                "total_usd": round(rag_with_cag_cost + storage_cost, 2),
                "cag_savings_usd": round(cag_savings, 2),
                "cag_savings_pct": round(cag_hit_rate * 100, 1),
                "cost_per_user_usd": round((rag_with_cag_cost + storage_cost) / daily_users / days_per_month, 4),
            },
            "crag_only": {
                "llm_cost_usd": round(crag_llm_cost, 2),
                "storage_cost_usd": round(storage_cost, 4),
                "total_usd": round(crag_llm_cost + storage_cost, 2),
                "vs_rag_savings_usd": round(rag_llm_cost - crag_llm_cost, 2),
                "cost_per_user_usd": round((crag_llm_cost + storage_cost) / daily_users / days_per_month, 4),
            },
        }

    cost_path = outputs_dir / "cost_analysis.json"
    with open(cost_path, "w") as f:
        json.dump(cost_analysis, f, indent=2)
    print(f"✅ Saved cost_analysis.json")

    # Print summary
    print("\n--- Cost Projection Summary ---")
    print(f"{'Scale':<20} {'RAG Only':>12} {'RAG+CAG':>12} {'CRAG':>12}")
    print("-" * 60)
    for scale_key, data in cost_analysis["scales"].items():
        users = data["daily_users"]
        rag = data["rag_only"]["total_usd"]
        rag_cag = data["rag_plus_cag"]["total_usd"]
        crag = data["crag_only"]["total_usd"]
        print(f"{users} users/day{'':<10} ${rag:>10.2f}  ${rag_cag:>10.2f}  ${crag:>10.2f}")


if __name__ == "__main__":
    asyncio.run(main())
