# 🏠 Prime Lands — Production-Grade Real Estate Intelligence Platform

> **Zuu Crew AI | AI Engineer Essentials Bootcamp | Mini Project 02 | Context Engineering**

A production-ready RAG system for Sri Lankan real estate, featuring hybrid search, corrective retrieval (CRAG), and cache-augmented generation (CAG) powered by Claude 3 Haiku and local FastEmbed embeddings.

## 🎯 Project Overview

**Prime Lands** is not just another RAG demo. This is a **production-grade** implementation featuring:

✅ **Hybrid Search**: Dense (FastEmbed BAAI/bge-small-en-v1.5) + Sparse (BM25) fusion  
✅ **CRAG Algorithm**: Per-document grading with corrective retrieval  
✅ **CAG Service**: Semantic caching with TTL and LRU eviction  
✅ **5 Chunking Strategies**: Semantic, Fixed, Sliding, Parent-Child, Late  
✅ **Cross-Encoder Reranking**: ms-marco-MiniLM for final precision  
✅ **RAGAS Evaluation**: Faithfulness, context recall, context precision  
✅ **Production Code**: Pydantic validation, structured logging, custom exceptions  

## 🏗️ Architecture

```
Playwright Crawler  ->  PropertyDocument  ->  JSONL Corpus
                               |
                               v
     5 Chunking Strategies  ->  Hybrid Indexing (Qdrant)
     Dense (FastEmbed) + Sparse (BM25)  ->  RRF Fusion
                               |
                               v
     CAG  ->  Cache Hit?  ->  Answer
               | miss
               v
     CRAG  ->  Grade Docs  ->  Route  ->  Generate (Claude)
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Docker** (for Qdrant)
- **API Keys**:
  - Anthropic API key (for Claude generation)
  - *(OpenAI key NOT required — local FastEmbed used for embeddings)*

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Isuruigi/prime-lands-rag.git
cd prime-lands-rag

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# 3. Install package in editable mode
pip install -e ".[dev]"

# 4. Install spaCy model
python -m spacy download en_core_web_sm

# 5. Install Playwright browsers
playwright install chromium

# 6. Set up environment variables
copy .env.example .env
# Edit .env and add your API keys

# 7. Start Qdrant vector database
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 ^
  -v "%cd%\qdrant_storage:/qdrant/storage" ^
  qdrant/qdrant

# 8. Verify Qdrant is running
curl http://localhost:6333/health
```

## 📦 Project Structure

```
prime_lands/
├── pyproject.toml           # Python packaging with dependencies
├── config.yaml              # All tunable parameters (hybrid API config)
├── .env                     # API keys (gitignored)
├── README.md
│
├── src/prime_lands/         # Installable package
│   ├── config.py           # Pydantic Settings validation
│   ├── logger.py           # Structured logging (loguru)
│   ├── exceptions.py       # Custom exception hierarchy
│   │
│   ├── crawler/
│   │   ├── models.py       # PropertyDocument Pydantic model
│   │   └── crawler.py      # PrimeLandsCrawler (async BFS)
│   │
│   ├── chunking/
│   │   ├── base.py         # AbstractChunker
│   │   ├── semantic.py     # Semantic chunking (spaCy)
│   │   ├── fixed.py        # Fixed-size (tiktoken)
│   │   ├── sliding.py      # Sliding window
│   │   ├── parent_child.py # Hierarchical chunking
│   │   └── late.py         # Late chunking (JINA)
│   │
│   ├── indexing/
│   │   └── qdrant_indexer.py  # Hybrid dense + sparse indexing
│   │
│   └── services/
│       ├── base.py         # AbstractIntelligenceService
│       ├── rag_service.py  # Basic RAG
│       ├── cag_service.py  # Cache-Augmented Generation
│       └── crag_service.py # Corrective RAG (CRAG paper)
│
├── notebooks/
│   ├── 01_crawl_primelands.ipynb      # Part 1: Web crawling
│   ├── 02_chunk_lab.ipynb             # Part 2: Chunking comparison
│   ├── 03_intelligence_layers.ipynb   # Part 3: RAG/CAG/CRAG
│   └── 04_performance_arena.ipynb     # Part 4: RAGAS evaluation
│
├── data/
│   ├── primelands_corpus.jsonl        # Crawled properties
│   └── chunks/                        # Chunked data per strategy
│
├── outputs/
│   ├── performance_comparison.csv  # RAG vs CRAG RAGAS scores
│   ├── chunking_comparison.csv     # 5 strategy comparison + RAGAS metrics
│   ├── cag_stats.json              # 100-query CAG simulation results
│   ├── crag_impact.csv             # 20-query CRAG vs RAG breakdown
│   └── cost_analysis.json          # 3-scale cost projection (bonus)
│
└── report/
    └── engineering_report.pdf
```

## 🔧 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Claude 3 Haiku | Fast, cost-effective generation |
| **Embeddings** | FastEmbed BAAI/bge-small-en-v1.5 | Local dense vectors (zero API cost) |
| **Vector DB** | Qdrant | Hybrid search (dense + sparse) |
| **Sparse Retrieval** | BM25 (FastEmbed) | Keyword-based precision |
| **Reranking** | ms-marco-MiniLM | Cross-encoder reranking |
| **Evaluation** | RAGAS | Faithfulness, context recall, precision |
| **Chunking** | spaCy, tiktoken | Multiple strategies |
| **Crawler** | Playwright | JS-rendered content handling |
| **Validation** | Pydantic | Type-safe configuration |
| **Logging** | Loguru | Structured logging |

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Properties crawled | 19 |
| CAG hit rate (100 queries) | 79% |
| CAG speedup factor | 22.75x |
| Best chunking faithfulness | 1.000 (fixed / parent_child / late) |
| Semantic chunking faithfulness | 0.963 |
| CRAG evaluations | 20 queries x 4 categories |

## 🎓 Usage Examples

### 1. Crawl Prime Lands Website

```python
from prime_lands.config import load_config
from prime_lands.crawler.crawler import PrimeLandsCrawler
from pathlib import Path

cfg = load_config()
crawler = PrimeLandsCrawler(cfg.crawler, data_dir=Path("data"))
properties = await crawler.crawl()
```

### 2. Index with Hybrid Search

```python
from prime_lands.indexing.qdrant_indexer import QdrantIndexer

indexer = QdrantIndexer(cfg)
await indexer.create_collection("primelands_semantic")
await indexer.index_chunks(chunks, collection="primelands_semantic")
```

### 3. Query with CRAG

```python
from prime_lands.services.crag_service import CRAGService

crag = CRAGService(cfg)
result = await crag.query("3 bedroom villa in Colombo under 50M")
print(result.answer)
```

## 🧪 Automated Pipeline Scripts

```bash
# Chunking + RAGAS evaluation across all 5 strategies
python run_chunking.py

# Index all 5 collections into Qdrant
python run_index_all.py

# Run 100-query CAG simulation
python run_cag_stats.py

# CRAG impact analysis (20 queries x 4 categories)
python run_crag_impact.py

# Full RAGAS benchmarking (RAG vs CRAG)
python run_performance.py
```

## 🔑 API Key Setup

1. **Anthropic** (Required for Claude): https://console.anthropic.com/
2. **OpenAI** (Optional, replaced by local FastEmbed): https://platform.openai.com/api-keys

Add to `.env`:
```bash
ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY is NOT required — local FastEmbed handles all embeddings
```

## 📝 Project Deliverables

- [x] Part 1: Production Crawler (15 pts)
- [x] Part 2: Chunking Lab (25 pts)
- [x] Part 3: Intelligence Layers (25 pts)
- [x] Part 4: Performance Arena (20 pts)
- [x] Engineering Report (15 pts)

**Total: 100 points (+5 bonus)**

## 📄 License

MIT License — Built for educational purposes as part of the Zuu Crew AI Engineer Essentials Bootcamp.
