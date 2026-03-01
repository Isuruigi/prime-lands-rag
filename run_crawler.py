"""
Standalone crawler script - Run this from terminal instead of Jupyter.
Avoids Windows asyncio issues in notebooks.
"""

import asyncio
import sys
from pathlib import Path

# Fix Windows asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from prime_lands.config import load_config
from prime_lands.crawler.crawler import PrimeLandsCrawler
from prime_lands.logger import setup_logger


async def main():
    """Run the crawler."""
    print("🚀 Starting Prime Lands Crawler...")
    print("=" * 60)
    
    # Setup
    setup_logger(level="INFO")
    cfg = load_config(Path("config.yaml"))
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Initialize crawler
    crawler = PrimeLandsCrawler(cfg.crawler, data_dir=data_dir)
    
    print(f"\n📋 Configuration:")
    print(f"  - Base URL: {cfg.crawler.base_url}")
    print(f"  - Max pages: {cfg.crawler.max_pages}")
    print(f"  - Rate limit: {cfg.crawler.rate_limit_seconds}s")
    print(f"  - Output: {data_dir / 'primelands_corpus.jsonl'}")
    
    # Crawl
    print(f"\n🔍 Starting crawl (this may take 5-15 minutes)...\n")
    properties = await crawler.crawl()
    
    # Results
    print(f"\n{'=' * 60}")
    print(f"✅ Crawl Complete!")
    print(f"{'=' * 60}")
    print(f"  Total properties: {len(properties)}")
    print(f"  Output file: {data_dir / 'primelands_corpus.jsonl'}")
    print(f"\n📊 Next steps:")
    print(f"  1. Review the crawled data")
    print(f"  2. Run chunking comparison (notebook 02)")
    print(f"  3. Continue with intelligence layers (notebook 03)")


if __name__ == "__main__":
    asyncio.run(main())
