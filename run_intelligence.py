
import asyncio
import sys
import json
import os
from pathlib import Path
from dotenv import load_dotenv

from prime_lands.config import load_config
from prime_lands.indexing.qdrant_indexer import QdrantIndexer
from prime_lands.services.rag_service import RAGService
from prime_lands.services.cag_service import CAGService
from prime_lands.services.crag_service import CRAGService
from prime_lands.chunking.base import Chunk
from prime_lands.logger import setup_logger, get_logger

log = get_logger(__name__)

async def main():
    setup_logger(level="INFO")
    
    # Paths
    root_dir = Path.cwd()
    data_dir = root_dir / "data"
    outputs_dir = root_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    load_dotenv(root_dir / ".env")
    cfg = load_config(root_dir / "config.yaml")
    
    # Load chunks
    chunks_file = data_dir / "chunks" / "semantic_chunks.jsonl"
    if not chunks_file.exists():
        log.error(f"Chunks file not found: {chunks_file}")
        return

    chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(Chunk(**json.loads(line)))
            
    log.info(f"Loaded {len(chunks)} chunks")

    # Indexing
    indexer = QdrantIndexer(cfg)
    collection_name = "primelands_semantic"
    
    log.info(f"Creating collection {collection_name}...")
    await indexer.create_collection(collection_name, force=True)
    
    log.info("Indexing chunks...")
    await indexer.index_chunks(chunks, collection_name, batch_size=32)
    log.info("Indexing complete")

    # Initialize Services
    rag_service = RAGService(cfg, indexer)
    cag_service = CAGService(cfg, rag_service)
    # crag_service = CRAGService(cfg, indexer) # CRAG might be slow/expensive, let's include it

    # Test Queries
    queries = [
        "What 3 bedroom properties are available in Colombo?",
        "Show me land for sale in Kottawa",
        "What are the amenities at The Seasons?",
    ]

    results = []

    print("\nRunning RAG tests...")
    for q in queries:
        res = await rag_service.query(q, collection_name=collection_name)
        print(f"Q: {q}\nA: {res.answer[:100]}...\n")
        results.append({
            "service": "RAG",
            "query": q,
            "latency": res.latency_ms,
            "cost": res.cost
        })

    print("\nRunning CAG tests (Caching)...")
    # Run same query twice to test cache
    cag_query = "What 3 bedroom properties are available in Colombo?"
    
    # First run (miss)
    res1 = await cag_service.query(cag_query, collection_name=collection_name)
    print(f"Run 1 (Miss): Latency={res1.latency_ms:.0f}ms, Cache Hit={res1.metadata.get('cache_hit')}")
    
    # Second run (hit)
    res2 = await cag_service.query(cag_query, collection_name=collection_name)
    print(f"Run 2 (Hit): Latency={res2.latency_ms:.0f}ms, Cache Hit={res2.metadata.get('cache_hit')}")

    # Save stats
    with open(outputs_dir / "cag_stats.json", "w") as f:
        json.dump(cag_service.get_stats(), f, indent=2)
        
    print(f"\nSaved results to {outputs_dir}")

if __name__ == "__main__":
    asyncio.run(main())
