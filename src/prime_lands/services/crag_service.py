"""
Corrective RAG (CRAG) service following the algorithm from the paper.
Document grading → routing → corrective retrieval → knowledge refinement.
"""

from __future__ import annotations
import os
import time
from typing import Any, Literal

from dotenv import load_dotenv
from anthropic import Anthropic

from prime_lands.config import PlatformConfig
from prime_lands.indexing.qdrant_indexer import QdrantIndexer
from prime_lands.services.base import AbstractIntelligenceService, QueryResult
from prime_lands.logger import get_logger

log = get_logger(__name__)
load_dotenv()


class CRAGService(AbstractIntelligenceService):
    """
    Corrective RAG implementation following Shi et al., 2023.
    
    Algorithm:
    1. Retrieve documents
    2. Grade each document for relevance (0.0-1.0)
    3. Route based on global confidence:
       - High confidence → Generate
       - Low confidence → Rewrite query + re-retrieve
       - Ambiguous → Refine knowledge + generate
    4. Generate answer with grounded context
    """
    
    def __init__(self, config: PlatformConfig, indexer: QdrantIndexer):
        self.cfg = config
        self.indexer = indexer
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Stats
        self.total_queries = 0
        self.corrections_triggered = 0
        self.total_latency = 0.0
        self.total_cost = 0.0
        
        log.info("Initialized CRAG Service with document grading")
    
    async def _grade_document(self, query: str, document: str) -> float:
        """
        Grade a document's relevance to query (0.0 = irrelevant, 1.0 = highly relevant).
        
        Uses Claude to assess relevance as a float score.
        """
        import re as _re
        prompt = f"""Rate how relevant this document is for answering the query.

Query: {query}

Document: {document[:1000]}

Reply with ONLY a decimal number between 0.0 and 1.0. No explanation, no text, just the number.
Examples of valid replies: 0.8   0.3   1.0   0.0

Score:"""
        
        response = self.anthropic.messages.create(
            model=self.cfg.llm.model,
            max_tokens=20,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        raw = response.content[0].text.strip()
        # Use regex to extract the first float/int in the response
        match = _re.search(r'\b(1\.0|0\.\d+|\d+\.\d+|[01])\b', raw)
        if match:
            score = float(match.group(1))
            score = max(0.0, min(1.0, score))
            log.debug(f"Document grade: {score:.2f} (raw: '{raw[:30]}')")
            return score
        
        log.warning(f"Failed to parse relevance score from '{raw[:50]}', defaulting to 0.5")
        return 0.5
    
    async def _rewrite_query(self, original_query: str) -> str:
        """Rewrite query for better retrieval."""
        prompt = f"""The following query did not retrieve relevant results. Rewrite it to be more specific and likely to find relevant information:

Original query: {original_query}

Rewritten query:"""
        
        response = self.anthropic.messages.create(
            model=self.cfg.llm.model,
            max_tokens=100,
            temperature=self.cfg.crag.rewrite_temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        
        rewritten = response.content[0].text.strip()
        log.info(f"Query rewrite: '{original_query}' → '{rewritten}'")
        return rewritten
    
    def _refine_knowledge(self, documents: list[str]) -> list[str]:
        """
        Knowledge refinement: strip irrelevant sentences from ambiguous docs.
        Simplified version - in production would use sentence-level grading.
        """
        # For now, just return the documents as-is
        # Full implementation would grade each sentence and filter
        return documents
    
    async def query(self, question: str, collection_name: str = "primelands_semantic", **kwargs) -> QueryResult:
        """
        Process CRAG query with document grading and corrective retrieval.
        
        Args:
            question: User question
            collection_name: Qdrant collection
            **kwargs: Additional parameters
            
        Returns:
            QueryResult with corrective RAG results
        """
        start_time = time.time()
        corrections = 0
        all_costs = []
        
        current_query = question
        
        for iteration in range(self.cfg.crag.max_correction_iterations + 1):
            # 1. Retrieve documents
            top_k = kwargs.get("top_k", self.cfg.retrieval.top_k)
            search_results = self.indexer.hybrid_search(
                query=current_query,
                collection_name=collection_name,
                top_k=top_k,
            )
            
            documents = [r["text"] for r in search_results]
            
            if not documents:
                log.warning(f"No documents retrieved for: {current_query}")
                break
            
            # 2. Grade each document
            log.info(f"Grading {len(documents)} documents...")
            grades = []
            for doc in documents[:5]:  # Grade top 5 to save cost
                grade = await self._grade_document(question, doc)
                grades.append(grade)
                log.debug(f"Document grade: {grade:.2f}")
            
            # 3. Calculate global confidence
            avg_grade = sum(grades) / len(grades) if grades else 0.0
            log.info(f"Average document grade: {avg_grade:.3f}")
            
            # 4. Route based on confidence
            if avg_grade >= self.cfg.crag.global_confidence_threshold:
                # High confidence → Generate
                log.info("✓ High confidence → Generating answer")
                relevant_docs = [
                    doc for doc, grade in zip(documents[:len(grades)], grades)
                    if grade >= self.cfg.crag.doc_relevance_threshold
                ]
                break
            
            elif iteration < self.cfg.crag.max_correction_iterations:
                # Low confidence → Corrective retrieval
                log.info(f"⚠ Low confidence (iter {iteration+1}) → Rewriting query")
                current_query = await self._rewrite_query(current_query)
                corrections += 1
                self.corrections_triggered += 1
            else:
                # Max iterations reached
                log.warning(f"Max correction iterations reached, using best available docs")
                relevant_docs = documents
                break
        
        # If we exited due to high confidence
        if 'relevant_docs' not in locals():
            relevant_docs = documents
        
        # Apply knowledge refinement if enabled
        if self.cfg.crag.knowledge_refinement:
            relevant_docs = self._refine_knowledge(relevant_docs)
        
        # 5. Generate final answer
        context_str = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(relevant_docs)])
        
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
            messages=[{"role": "user", "content": prompt}]
        )
        
        answer = response.content[0].text
        
        # Calculate cost
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
            contexts=relevant_docs,
            metadata={
                "service": "crag",
                "model": self.cfg.llm.model,
                "corrections_applied": corrections,
                "avg_document_grade": avg_grade,
                "retrieved_chunks": len(documents),
                "relevant_chunks": len(relevant_docs),
                "final_query": current_query,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            latency_ms=latency_ms,
            cost=cost,
        )
    
    def get_stats(self) -> dict[str, Any]:
        """Get CRAG service statistics."""
        avg_latency = self.total_latency / self.total_queries if self.total_queries > 0 else 0
        correction_rate = (self.corrections_triggered / self.total_queries * 100) if self.total_queries > 0 else 0
        
        return {
            "service": self.service_name,
            "total_queries": self.total_queries,
            "corrections_triggered": self.corrections_triggered,
            "correction_rate_pct": correction_rate,
            "avg_latency_ms": avg_latency,
            "total_cost_usd": self.total_cost,
            "avg_cost_per_query": self.total_cost / self.total_queries if self.total_queries > 0 else 0,
        }
    
    @property
    def service_name(self) -> str:
        return "crag"
