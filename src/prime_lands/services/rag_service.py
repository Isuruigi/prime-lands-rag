"""
Basic RAG service with Claude generation.
Retrieve → Generate with context.
"""

from __future__ import annotations
import os
import time
from typing import Any

from dotenv import load_dotenv
from anthropic import Anthropic

from prime_lands.config import PlatformConfig
from prime_lands.indexing.qdrant_indexer import QdrantIndexer
from prime_lands.services.base import AbstractIntelligenceService, QueryResult
from prime_lands.logger import get_logger

log = get_logger(__name__)
load_dotenv()


class RAGService(AbstractIntelligenceService):
    """
    Basic RAG implementation with Claude.
    
    Flow:
    1. Retrieve top-k relevant chunks (hybrid search)
    2. Generate answer using Claude with retrieved context
    """
    
    def __init__(self, config: PlatformConfig, indexer: QdrantIndexer):
        self.cfg = config
        self.indexer = indexer
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Stats tracking
        self.total_queries = 0
        self.total_latency = 0.0
        self.total_cost = 0.0
        
        log.info("Initialized RAG Service with Claude")
    
    async def query(self, question: str, collection_name: str = "primelands_semantic", **kwargs) -> QueryResult:
        """
        Process RAG query with Claude generation.
        
        Args:
            question: User question
            collection_name: Qdrant collection to search
            **kwargs: Additional parameters
            
        Returns:
            QueryResult with answer and context
        """
        start_time = time.time()
        
        # 1. Retrieve contexts
        top_k = kwargs.get("top_k", self.cfg.retrieval.top_k)
        search_results = self.indexer.hybrid_search(
            query=question,
            collection_name=collection_name,
            top_k=top_k,
        )
        
        contexts = [r["text"] for r in search_results]
        
        # 2. Generate with Claude
        context_str = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""Context information:
{context_str}

Based on the above context, please answer the following question:
{question}

If the context doesn't contain relevant information, say so."""
        
        response = self.anthropic.messages.create(
            model=self.cfg.llm.model,
            max_tokens=self.cfg.llm.max_tokens,
            temperature=self.cfg.llm.temperature,
            system=self.cfg.rag.system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        answer = response.content[0].text
        
        # Calculate cost (Claude 3.5 Haiku pricing: ~$0.80/MTok input, $4/MTok output)
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = (input_tokens / 1_000_000 * 0.80) + (output_tokens / 1_000_000 * 4.0)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Update stats
        self.total_queries += 1
        self.total_latency += latency_ms
        self.total_cost += cost
        
        return QueryResult(
            answer=answer,
            contexts=contexts,
            metadata={
                "service": "rag",
                "model": self.cfg.llm.model,
                "retrieved_chunks": len(contexts),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            latency_ms=latency_ms,
            cost=cost,
        )
    
    def get_stats(self) -> dict[str, Any]:
        """Get RAG service statistics."""
        avg_latency = self.total_latency / self.total_queries if self.total_queries > 0 else 0
        
        return {
            "service": self.service_name,
            "total_queries": self.total_queries,
            "avg_latency_ms": avg_latency,
            "total_cost_usd": self.total_cost,
            "avg_cost_per_query": self.total_cost / self.total_queries if self.total_queries > 0 else 0,
        }
    
    @property
    def service_name(self) -> str:
        return "rag"
