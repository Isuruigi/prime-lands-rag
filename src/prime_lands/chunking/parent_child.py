"""
Parent-child hierarchical chunking.
Indexes small child chunks but retrieves with full parent context.
"""

from __future__ import annotations
import hashlib
from typing import Any

import tiktoken

from prime_lands.chunking.base import AbstractChunker, Chunk
from prime_lands.config import ParentChildConfig
from prime_lands.logger import get_logger

log = get_logger(__name__)


class ParentChildChunker(AbstractChunker):
    """
    Hierarchical parent-child chunking.
    
    Strategy:
    - Split text into large parent chunks
    - Split each parent into smaller child chunks
    - At retrieval: find child chunk, return full parent context
    
    This balances precise retrieval (small chunks) with rich context (large chunks).
    """
    
    def __init__(self, config: ParentChildConfig):
        self.cfg = config
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def chunk(self, text: str, source_id: str = "") -> list[Chunk]:
        """
        Create parent-child chunk hierarchy.
        
        Args:
            text: Input text
            source_id: Source document ID
            
        Returns:
            List of child chunks (with parent_id in metadata)
        """
        tokens = self.encoding.encode(text)
        all_chunks = []
        
        # Create parent chunks
        parent_idx = 0
        start_idx = 0
        
        while start_idx < len(tokens):
            parent_end = min(start_idx + self.cfg.parent_size, len(tokens))
            parent_tokens = tokens[start_idx:parent_end]
            parent_text = self.encoding.decode(parent_tokens)
            parent_id = f"{source_id}_p{parent_idx}"
            
            # Create child chunks within this parent
            child_start = 0
            child_idx = 0
            
            while child_start < len(parent_tokens):
                child_end = min(child_start + self.cfg.child_size, len(parent_tokens))
                child_tokens = parent_tokens[child_start:child_end]
                child_text = self.encoding.decode(child_tokens)
                
                chunk_id = hashlib.md5(child_text.encode()).hexdigest()[:12]
                all_chunks.append(Chunk(
                    text=child_text,
                    chunk_id=chunk_id,
                    source_id=source_id,
                    metadata={
                        "strategy": "parent_child",
                        "parent_id": parent_id,
                        "parent_text": parent_text if self.cfg.link_children else "",
                        "child_index": child_idx,
                        "token_count": len(child_tokens),
                        "parent_size": self.cfg.parent_size,
                        "child_size": self.cfg.child_size,
                    }
                ))
                
                child_start += self.cfg.child_size
                child_idx += 1
            
            start_idx += self.cfg.parent_size
            parent_idx += 1
        
        return all_chunks
    
    def get_stats(self, chunks: list[Chunk]) -> dict[str, Any]:
        """Compute statistics for parent-child chunks."""
        if not chunks:
            return {}
        
        # Count unique parents
        parent_ids = set(c.metadata.get("parent_id", "") for c in chunks)
        token_counts = [c.metadata.get("token_count", 0) for c in chunks]
        
        return {
            "strategy": self.strategy_name,
            "total_child_chunks": len(chunks),
            "total_parent_chunks": len(parent_ids),
            "avg_children_per_parent": len(chunks) / len(parent_ids) if parent_ids else 0,
            "avg_child_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
            "parent_size": self.cfg.parent_size,
            "child_size": self.cfg.child_size,
        }
    
    @property
    def strategy_name(self) -> str:
        return "parent_child"
