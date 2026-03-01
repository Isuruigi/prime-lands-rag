"""
Abstract base class for all chunking strategies.
Enforces consistent interface across implementations.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """
    Unified chunk representation across all strategies.
    
    Attributes:
        text: The chunk content
        metadata: Strategy-specific metadata (start/end positions, parent ID, etc.)
        chunk_id: Unique identifier for this chunk
        source_id: ID of source document
    """
    text: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_id: str = ""
    source_id: str = ""


class AbstractChunker(ABC):
    """
    Base class for all chunking strategies.
    
    All implementations must provide:
    - chunk(): split text into Chunk objects
    - get_stats(): return chunking statistics
    """
    
    @abstractmethod
    def chunk(self, text: str, source_id: str = "") -> list[Chunk]:
        """
        Split text into chunks according to strategy.
        
        Args:
            text: Input text to chunk
            source_id: Identifier for source document
            
        Returns:
            List of Chunk objects
        """
        pass
    
    @abstractmethod
    def get_stats(self, chunks: list[Chunk]) -> dict[str, Any]:
        """
        Compute statistics for a list of chunks.
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Dictionary with stats (avg length, count, etc.)
        """
        pass
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return human-readable name of this strategy."""
        pass
