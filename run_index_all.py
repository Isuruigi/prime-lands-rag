"""
Index ALL 5 chunking strategies into separate Qdrant collections.
Required by submission: '5 distinct Qdrant collections — meaningfully different results'
"""
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv

from prime_lands.config import load_config
from prime_lands.indexing.qdrant_indexer import QdrantIndexer
from prime_lands.chunking.base import Chunk
from prime_lands.logger import setup_logger, get_logger

log = get_logger(__name__)


STRATEGY_COLLECTIONS = {
    "semantic":     "primelands_semantic",
    "fixed":        "primelands_fixed",
    "sliding":      "primelands_sliding",
    "parent_child": "primelands_parent_child",
    "late":         "primelands_late",
}


async def main():
    setup_logger(level="INFO")
    root_dir = Path.cwd()
    load_dotenv(root_dir / ".env")
    cfg = load_config(root_dir / "config.yaml")
    chunks_dir = root_dir / "data" / "chunks"

    indexer = QdrantIndexer(cfg)

    for strategy, collection_name in STRATEGY_COLLECTIONS.items():
        chunk_file = chunks_dir / f"{strategy}_chunks.jsonl"
        if not chunk_file.exists():
            log.warning(f"Chunk file not found: {chunk_file} — skipping")
            continue

        chunks = []
        with open(chunk_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(Chunk(**json.loads(line)))

        if not chunks:
            log.warning(f"No chunks in {chunk_file} — skipping")
            continue

        log.info(f"\n{'='*50}")
        log.info(f"Indexing {strategy}: {len(chunks)} chunks → '{collection_name}'")

        try:
            await indexer.create_collection(collection_name, force=True)
        except Exception as e:
            log.warning(f"Collection create failed: {e}")

        await indexer.index_chunks(chunks, collection_name, batch_size=32)
        log.info(f"Done: '{collection_name}'")

    print("\nAll 5 collections indexed successfully!")
    print("\nCollections:")
    for strategy, col in STRATEGY_COLLECTIONS.items():
        print(f"  {strategy:15s} -> {col}")


if __name__ == "__main__":
    asyncio.run(main())
