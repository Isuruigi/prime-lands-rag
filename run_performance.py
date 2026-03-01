
import os
import sys
import json
import asyncio
import pandas as pd
from pathlib import Path
from typing import Any, List, Optional
from dotenv import load_dotenv

import anthropic as anthropic_sdk
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_community.embeddings import FastEmbedEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, context_recall, context_precision
from datasets import Dataset


class AnthropicLLM(LLM):
    """Minimal LangChain-compatible wrapper around the Anthropic SDK for RAGAS."""
    model_name: str = "claude-3-haiku-20240307"
    max_tokens: int = 1024
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

from prime_lands.config import load_config
from prime_lands.indexing.qdrant_indexer import QdrantIndexer
from prime_lands.services.rag_service import RAGService
from prime_lands.services.cag_service import CAGService
from prime_lands.services.crag_service import CRAGService
from prime_lands.logger import setup_logger

async def main():
    setup_logger(level="INFO")
    
    # Paths
    root_dir = Path.cwd()
    outputs_dir = root_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    load_dotenv(root_dir / ".env")
    cfg = load_config(root_dir / "config.yaml")

    # Initialize Services
    indexer = QdrantIndexer(cfg)
    rag_service = RAGService(cfg, indexer)
    crag_service = CRAGService(cfg, indexer)
    
    collection_name = "primelands_semantic"

    # Test Data
    test_cases = [
        {
            "question": "What 3 bedroom properties are available?",
            "ground_truth": "Multiple 3-bedroom properties are available including houses, apartments, and villas with various amenities."
        },
        {
            "question": "What are the typical amenities in luxury properties?",
            "ground_truth": "Luxury properties typically include features like swimming pools, gyms, security systems, parking spaces, and modern finishes."
        },
        {
            "question": "Are there properties near Colombo city center?",
            "ground_truth": "Yes, there are various residential and commercial properties available in and around Colombo city center."
        },
        {
             "question": "What is the price range for apartments?",
             "ground_truth": "Apartment prices vary based on location, size, and amenities, ranging from budget-friendly to luxury options."
        },
        {
             "question": "Do any properties have ocean views?",
             "ground_truth": "Some properties, particularly coastal villas and high-rise apartments, offer ocean or sea views."
        }
    ]

    # Configure RAGAS with Custom Models (Avoid OpenAI)
    # Use direct Anthropic SDK wrapper for LLM evaluation
    eval_llm = AnthropicLLM(model_name="claude-3-haiku-20240307")
    # Use FastEmbed for evaluation embeddings
    eval_embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # 1. Evaluate RAG
    print("\n--------------------------------------------------")
    print("Evaluating RAG Service...")
    rag_results = []
    for case in test_cases:
        res = await rag_service.query(case["question"], collection_name=collection_name)
        rag_results.append({
            "question": case["question"],
            "answer": res.answer,
            "contexts": res.contexts,
            "ground_truth": case["ground_truth"],
            "latency_ms": res.latency_ms,
            "cost": res.cost,
        })
    
    rag_dataset = Dataset.from_dict({
        "question": [r["question"] for r in rag_results],
        "answer": [r["answer"] for r in rag_results],
        "contexts": [r["contexts"] for r in rag_results],
        "ground_truth": [r["ground_truth"] for r in rag_results],
    })

    print("Running RAGAS metrics for RAG...")
    try:
        rag_scores = evaluate(
            rag_dataset,
            metrics=[faithfulness, context_recall, context_precision],
            llm=eval_llm,
            embeddings=eval_embeddings,
        )
        print("RAG Scores:", rag_scores)
    except Exception as e:
        print(f"RAG evaluation failed: {e}")
        rag_scores = {}

    # 2. Evaluate CRAG
    print("\n--------------------------------------------------")
    print("Evaluating CRAG Service...")
    crag_results = []
    for case in test_cases:
        res = await crag_service.query(case["question"], collection_name=collection_name)
        crag_results.append({
            "question": case["question"],
            "answer": res.answer,
            "contexts": res.contexts,
            "ground_truth": case["ground_truth"],
            "latency_ms": res.latency_ms,
            "cost": res.cost,
            "corrections": res.metadata.get("corrections_applied", 0),
        })

    crag_dataset = Dataset.from_dict({
        "question": [r["question"] for r in crag_results],
        "answer": [r["answer"] for r in crag_results],
        "contexts": [r["contexts"] for r in crag_results],
        "ground_truth": [r["ground_truth"] for r in crag_results],
    })

    print("Running RAGAS metrics for CRAG...")
    try:
        crag_scores = evaluate(
            crag_dataset,
            metrics=[faithfulness, context_recall, context_precision],
            llm=eval_llm,
            embeddings=eval_embeddings,
        )
        print("CRAG Scores:", crag_scores)
    except Exception as e:
        print(f"CRAG evaluation failed: {e}")
        crag_scores = {}

    # Compare
    comparison = [
        {
            "Service": "RAG",
            "Faithfulness": rag_scores["faithfulness"],
            "Context Recall": rag_scores["context_recall"],
            "Context Precision": rag_scores["context_precision"],
            "Latency (ms)": sum(r["latency_ms"] for r in rag_results) / len(rag_results),
            "Cost ($)": sum(r["cost"] for r in rag_results),
        },
        {
            "Service": "CRAG",
            "Faithfulness": crag_scores["faithfulness"],
            "Context Recall": crag_scores["context_recall"],
            "Context Precision": crag_scores["context_precision"],
            "Latency (ms)": sum(r["latency_ms"] for r in crag_results) / len(crag_results),
            "Cost ($)": sum(r["cost"] for r in crag_results),
        }
    ]
    
    df = pd.DataFrame(comparison)
    print("\nComparison:\n", df)
    df.to_csv(outputs_dir / "performance_comparison.csv", index=False)
    print(f"\nSaved comparison to {outputs_dir}")

if __name__ == "__main__":
    asyncio.run(main())
