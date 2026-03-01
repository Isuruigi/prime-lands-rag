"""
Sliding window chunking with configurable step size.
Generates overlapping windows for dense coverage.
"""

from __future__ import annotations
import hashlib
from typing import Any

import tiktoken

from prime_lands.chunking.base import AbstractChunker, Chunk
from prime_lands.config import SlidingChunkConfig
from prime_lands.logger import get_logger

log = get_logger(__name__)


class SlidingChunker(AbstractChunker):
    """
    Sliding window chunking.
    
    Creates overlapping windows that slide across the text.
    More overlap than fixed chunking = better retrieval recall.
    """
    
    def __init__(self, config: SlidingChunkConfig):
        self.cfg = config
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def chunk(self, text: str, source_id: str = "") -> list[Chunk]:
        """
        Create sliding windows over text.
        
        Args:
            text: Input text
            source_id: Source document ID
            
        Returns:
            List of overlapping window chunks
        """
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= self.cfg.window_size:
            chunk_id = hashlib.md5(text.encode()).hexdigest()[:12]
            return [Chunk(
                text=text,
                chunk_id=chunk_id,
                source_id=source_id,
                metadata={
                    "strategy": "sliding",
                    "token_count": len(tokens),
                    "window_size": self.cfg.window_size,
                    "step_size": self.cfg.step_size,
                }
            )]
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            end_idx = min(start_idx + self.cfg.window_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                source_id=source_id,
                metadata={
                    "strategy": "sliding",
                    "token_count": len(chunk_tokens),
                    "start_token": start_idx,
                    "end_token": end_idx,
                    "window_size": self.cfg.window_size,
                    "step_size": self.cfg.step_size,
                }
            ))
            
            # Slide window by step_size
            start_idx += self.cfg.step_size
            
            # Stop if remaining text < half window (avoid tiny tail chunks)
            if end_idx >= len(tokens):
                break
        
        return chunks
    
    def get_stats(self, chunks: list[Chunk]) -> dict[str, Any]:
        """Compute statistics for sliding window chunks."""
        if not chunks:
            return {}
        
        token_counts = [c.metadata.get("token_count", 0) for c in chunks]
        char_lengths = [len(c.text) for c in chunks]
        
        # Calculate overlap percentage
        if len(chunks) > 1:
            overlap_tokens = self.cfg.window_size - self.cfg.step_size
            overlap_pct = (overlap_tokens / self.cfg.window_size) * 100
        else:
            overlap_pct = 0
        
        return {
            "strategy": self.strategy_name,
            "total_chunks": len(chunks),
            "avg_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
            "avg_chars": sum(char_lengths) / len(char_lengths),
            "total_tokens": sum(token_counts),
            "window_size": self.cfg.window_size,
            "step_size": self.cfg.step_size,
            "overlap_percentage": overlap_pct,
        }
    
    @property
    def strategy_name(self) -> str:
        return "sliding"
