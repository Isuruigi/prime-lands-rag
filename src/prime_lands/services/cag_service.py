"""
Cache-Augmented Generation (CAG) service.
Semantic caching with TTL and LRU eviction.
"""

from __future__ import annotations
import hashlib
import time
from collections import OrderedDict
from typing import Any
from datetime import datetime, timedelta

from sentence_transformers import SentenceTransformer, util

from prime_lands.config import PlatformConfig
from prime_lands.services.rag_service import RAGService
from prime_lands.services.base import QueryResult
from prime_lands.logger import get_logger

log = get_logger(__name__)


class CacheEntry:
    """Cache entry with TTL and metadata."""
    def __init__(self, query: str, result: QueryResult, ttl_hours: int, is_faq: bool = False):
        self.query = query
        self.result = result
        self.created_at = datetime.now()
        self.ttl_hours = ttl_hours
        self.is_faq = is_faq
        self.hit_count = 0
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        expires_at = self.created_at + timedelta(hours=self.ttl_hours)
        return datetime.now() > expires_at


class CAGService:
    """
    Cache-Augmented Generation service.
    
    Features:
    - Semantic similarity matching (not exact string match)
    - TTL with LRU eviction
    - FAQ prioritization (longer TTL)
    - Hit rate tracking
    """
    
    def __init__(self, config: PlatformConfig, rag_service: RAGService):
        self.cfg = config
        self.rag = rag_service
        
        # Semantic similarity model for cache matching
        self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # LRU cache using OrderedDict
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_queries = 0
        
        log.info(f"Initialized CAG Service (similarity threshold: {config.cag.similarity_threshold})")
    
    def _find_similar_query(self, query: str) -> CacheEntry | None:
        """
        Find cached query with similarity above threshold.
        
        Args:
            query: Input query
            
        Returns:
            CacheEntry if similar cached query found, None otherwise
        """
        if not self.cache:
            return None
        
        query_embedding = self.similarity_model.encode(query, convert_to_tensor=True)
        
        for cache_key, entry in self.cache.items():
            # Skip expired entries
            if entry.is_expired():
                continue
            
            cached_embedding = self.similarity_model.encode(entry.query, convert_to_tensor=True)
            similarity = util.cos_sim(query_embedding, cached_embedding).item()
            
            if similarity >= self.cfg.cag.similarity_threshold:
                log.info(f"Cache HIT (similarity: {similarity:.3f}): '{query}' → '{entry.query}'")
                entry.hit_count += 1
                # Move to end (LRU)
                self.cache.move_to_end(cache_key)
                return entry
        
        return None
    
    def _evict_if_needed(self) -> None:
        """Evict oldest entry if cache is full."""
        if len(self.cache) >= self.cfg.cag.max_history_size:
            # Remove oldest (first item in OrderedDict)
            oldest_key = next(iter(self.cache))
            removed = self.cache.pop(oldest_key)
            log.info(f"Evicted cache entry: '{removed.query}' (hits: {removed.hit_count})")
    
    async def query(self, question: str, is_faq: bool = False, **kwargs) -> QueryResult:
        """
        Process query with caching.
        
        Args:
            question: User question
            is_faq: Mark as FAQ for longer TTL
            **kwargs: Additional parameters
            
        Returns:
            QueryResult (from cache or fresh generation)
        """
        start_time = time.time()
        self.total_queries += 1
        
        # 1. Check cache
        cached = self._find_similar_query(question)
        
        if cached:
            self.cache_hits += 1
            result = cached.result
            result.metadata["cache_hit"] = True
            result.metadata["cache_hit_count"] = cached.hit_count
            result.metadata["cached_query"] = cached.query
            result.latency_ms = (time.time() - start_time) * 1000  # Cache lookup time
            result.cost = 0.0  # No API cost for cached result
            log.info(f"✓ Cache HIT: {self.cache_hits}/{self.total_queries} ({self.get_hit_rate():.1f}%)")
            return result
        
        # 2. Cache miss → RAG query
        self.cache_misses += 1
        log.info(f"Cache MISS: Querying RAG service...")
        
        result = await self.rag.query(question, **kwargs)
        
        # 3. Store in cache
        ttl = self.cfg.cag.cache_ttl_hours
        if is_faq:
            ttl *= self.cfg.cag.faq_ttl_multiplier
        
        cache_key = hashlib.md5(question.encode()).hexdigest()
        self.cache[cache_key] = CacheEntry(question, result, ttl, is_faq)
        result.metadata["cache_hit"] = False
        
        # Evict if needed
        self._evict_if_needed()
        
        log.info(f"Cache MISS: {self.cache_misses}/{self.total_queries} (stored with TTL: {ttl}h)")
        return result
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate percentage."""
        if self.total_queries == 0:
            return 0.0
        return (self.cache_hits / self.total_queries) * 100
    
    def get_stats(self) -> dict[str, Any]:
        """Get CAG service statistics."""
        # Count active (non-expired) entries
        active_entries = sum(1 for e in self.cache.values() if not e.is_expired())
        faq_entries = sum(1 for e in self.cache.values() if e.is_faq and not e.is_expired())
        
        return {
            "service": "cag",
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_pct": self.get_hit_rate(),
            "active_cache_entries": active_entries,
            "faq_entries": faq_entries,
            "cache_size": len(self.cache),
            "max_cache_size": self.cfg.cag.max_history_size,
        }
    
    @property
    def service_name(self) -> str:
        return "cag"
