"""
CAG 100-query simulation per the submission checklist spec.

Spec (from miniproject2_guide.md Part 4B):
- Simulate 100 queries (60% FAQ repeats from pool of 5, 40% mixed)
- Track: hit_rate_pct, faq_hits, history_hits, misses
- Calculate: avg latency by tier (faq hit / history hit / miss)
- Calculate: API calls avoided, estimated cost savings ($0.002 per avoided call)
- Calculate: speedup factor (miss_latency / hit_latency)
- Save to cag_stats.json
"""

import asyncio
import json
import random
import time
from pathlib import Path
from dotenv import load_dotenv

from prime_lands.config import load_config
from prime_lands.indexing.qdrant_indexer import QdrantIndexer
from prime_lands.services.rag_service import RAGService
from prime_lands.services.cag_service import CAGService
from prime_lands.logger import setup_logger


# ── Query pools ─────────────────────────────────────────────────────────────

FAQ_POOL = [
    "What 3 bedroom properties are available?",
    "Show me luxury villas for sale.",
    "Are there apartments near Colombo?",
    "What is the price range for 2 bedroom apartments?",
    "Properties with swimming pools.",
]

MIXED_POOL = [
    "Properties in Negombo with ocean views.",
    "Commercial properties for rent.",
    "What amenities are available in Gampaha?",
    "Affordable houses under 15 million LKR.",
    "Properties near schools in Colombo.",
    "Luxury penthouses with parking.",
    "Land for sale in Kandy.",
    "What properties have gardens?",
    "Are there serviced apartments available?",
    "Which properties are suitable for investment?",
    "Show me properties in Thalawathugoda.",
    "What is available in Malabe?",
    "Properties with gym and pool facilities.",
    "Houses with more than 4 bedrooms.",
    "What are the newest listings?",
    "Properties near Colombo city centre.",
    "How many properties have sea views?",
    "What condos are for sale?",
    "Properties under construction.",
    "Which properties have rooftop terraces?",
]


def build_query_sequence(n: int = 100, faq_ratio: float = 0.6, seed: int = 42) -> list[dict]:
    """
    Build a 100-query sequence: 60% FAQ repeats, 40% mixed.
    Returns a list of dicts: {"query": str, "is_faq": bool}
    """
    random.seed(seed)
    faq_count = int(n * faq_ratio)
    mixed_count = n - faq_count

    faq_queries  = [{"query": random.choice(FAQ_POOL),   "is_faq": True}  for _ in range(faq_count)]
    mixed_queries = [{"query": random.choice(MIXED_POOL), "is_faq": False} for _ in range(mixed_count)]

    combined = faq_queries + mixed_queries
    random.shuffle(combined)
    return combined


async def main():
    setup_logger(level="WARNING")   # quiet logs during bulk run
    root_dir = Path.cwd()
    load_dotenv(root_dir / ".env")
    cfg = load_config(root_dir / "config.yaml")

    indexer     = QdrantIndexer(cfg)
    rag_service = RAGService(cfg, indexer)
    cag_service = CAGService(cfg, rag_service)

    collection_name = "primelands_semantic"

    queries = build_query_sequence(100)
    print(f"\nRunning {len(queries)}-query CAG simulation "
          f"(60% FAQ repeats / 40% mixed)...")
    print("=" * 60)

    # Per-tier latency tracking
    latencies: dict[str, list[float]] = {
        "faq_hit":     [],
        "history_hit": [],
        "miss":        [],
    }
    faq_hits = 0
    history_hits = 0
    misses = 0

    for i, item in enumerate(queries):
        q       = item["query"]
        is_faq  = item["is_faq"]

        t0 = time.time()
        result = await cag_service.query(q, is_faq=is_faq,
                                         collection_name=collection_name)
        elapsed_ms = (time.time() - t0) * 1000

        hit  = result.metadata.get("cache_hit", False)
        tier: str

        if hit and is_faq:
            faq_hits += 1
            tier = "faq_hit"
        elif hit:
            history_hits += 1
            tier = "history_hit"
        else:
            misses += 1
            tier = "miss"

        latencies[tier].append(elapsed_ms)

        label = "FAQ-HIT " if (hit and is_faq) else ("HIT     " if hit else "MISS    ")
        print(f"[{i+1:3d}] {label} | {q[:55]}")

    # ── Summary stats ────────────────────────────────────────────────────────
    total = len(queries)
    total_hits = faq_hits + history_hits

    def avg(lst):
        return round(sum(lst) / len(lst), 1) if lst else 0.0

    avg_faq_hit_ms      = avg(latencies["faq_hit"])
    avg_history_hit_ms  = avg(latencies["history_hit"])
    avg_miss_ms         = avg(latencies["miss"])
    avg_hit_ms          = avg(latencies["faq_hit"] + latencies["history_hit"])

    api_calls_avoided   = total_hits
    cost_per_call_usd   = 0.002          # per guide spec
    cost_savings_usd    = round(api_calls_avoided * cost_per_call_usd, 4)
    speedup_factor      = round(avg_miss_ms / avg_hit_ms, 2) if avg_hit_ms > 0 else 0.0

    stats = {
        "service":              "cag",
        "simulation": {
            "total_queries":    total,
            "faq_pool_size":    len(FAQ_POOL),
            "faq_ratio_pct":    60.0,
            "mixed_ratio_pct":  40.0,
        },
        "cache_performance": {
            "faq_hits":         faq_hits,
            "history_hits":     history_hits,
            "total_hits":       total_hits,
            "misses":           misses,
            "hit_rate_pct":     round((total_hits / total) * 100, 1),
            "faq_hit_rate_pct": round((faq_hits    / total) * 100, 1),
        },
        "latency_by_tier_ms": {
            "faq_hit":          avg_faq_hit_ms,
            "history_hit":      avg_history_hit_ms,
            "miss":             avg_miss_ms,
            "avg_hit":          avg_hit_ms,
        },
        "economics": {
            "api_calls_avoided":        api_calls_avoided,
            "cost_per_avoided_call_usd": cost_per_call_usd,
            "estimated_cost_savings_usd": cost_savings_usd,
            "speedup_factor":           speedup_factor,
        },
        "cache_state": {
            "active_cache_entries": sum(1 for e in cag_service.cache.values() if not e.is_expired()),
            "faq_entries":          sum(1 for e in cag_service.cache.values() if e.is_faq and not e.is_expired()),
            "max_cache_size":       cfg.cag.max_history_size,
        },
    }

    # Print summary
    print("\n" + "=" * 60)
    print("CAG 100-QUERY SIMULATION RESULTS")
    print("=" * 60)
    print(f"Total queries     : {total}")
    print(f"FAQ hits          : {faq_hits}  ({stats['cache_performance']['faq_hit_rate_pct']:.1f}%)")
    print(f"History hits      : {history_hits}")
    print(f"Misses            : {misses}")
    print(f"Overall hit rate  : {stats['cache_performance']['hit_rate_pct']:.1f}%")
    print(f"")
    print(f"Avg latency (FAQ hit):      {avg_faq_hit_ms:.1f} ms")
    print(f"Avg latency (History hit):  {avg_history_hit_ms:.1f} ms")
    print(f"Avg latency (Miss):         {avg_miss_ms:.1f} ms")
    print(f"Speedup factor  : {speedup_factor}x")
    print(f"API calls avoided: {api_calls_avoided}")
    print(f"Est. cost savings: ${cost_savings_usd:.4f}")

    out_path = root_dir / "outputs" / "cag_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
