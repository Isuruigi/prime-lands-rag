"""
Abstract base class for intelligence services.
Enforces consistent interface across RAG, CAG, and CRAG implementations.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel


class QueryResult(BaseModel):
    """
    Unified result format across all intelligence services.
    
    Attributes:
        answer: Generated answer text
        contexts: Retrieved context chunks
        metadata: Service-specific metadata (cache hit, corrections, etc.)
        latency_ms: Query processing time
        cost: API cost in USD (if applicable)
    """
    answer: str
    contexts: list[str] = []
    metadata: dict[str, Any] = {}
    latency_ms: float = 0.0
    cost: float = 0.0


class AbstractIntelligenceService(ABC):
    """
    Base class for all intelligence services (RAG, CAG, CRAG).
    
    All implementations must provide:
    - query(): Process a query and return result
    - get_stats(): Return service statistics
    """
    
    @abstractmethod
    async def query(self, question: str, **kwargs) -> QueryResult:
        """
        Process a query and return answer with context.
        
        Args:
            question: User question
            **kwargs: Service-specific parameters
            
        Returns:
            QueryResult with answer and metadata
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """
        Get service statistics.
        
        Returns:
            Dictionary with stats (cache hit rate, avg latency, etc.)
        """
        pass
    
    @property
    @abstractmethod
    def service_name(self) -> str:
        """Return human-readable name of this service."""
        pass
