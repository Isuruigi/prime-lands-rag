"""
Fixed-size chunking with overlap using tiktoken tokenizer.
Simple, predictable, works well for most RAG use cases.
"""

from __future__ import annotations
import hashlib
from typing import Any

import tiktoken

from prime_lands.chunking.base import AbstractChunker, Chunk
from prime_lands.config import FixedChunkConfig
from prime_lands.logger import get_logger

log = get_logger(__name__)


class FixedChunker(AbstractChunker):
    """
    Fixed-size chunking with overlap.
    
    Uses tiktoken (OpenAI's tokenizer) for accurate token counting.
    Overlap ensures context continuity across chunk boundaries.
    """
    
    def __init__(self, config: FixedChunkConfig):
        self.cfg = config
        # Use cl100k_base (GPT-4 tokenizer) - works for both OpenAI and Claude
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def chunk(self, text: str, source_id: str = "") -> list[Chunk]:
        """
        Chunk text into fixed-size pieces with overlap.
        
        Args:
            text: Input text
            source_id: Source document ID
            
        Returns:
            List of fixed-size chunks with overlap
        """
        # Tokenize entire text
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= self.cfg.chunk_size:
            # Text fits in one chunk
            chunk_id = hashlib.md5(text.encode()).hexdigest()[:12]
            return [Chunk(
                text=text,
                chunk_id=chunk_id,
                source_id=source_id,
                metadata={
                    "strategy": "fixed",
                    "token_count": len(tokens),
                    "chunk_size": self.cfg.chunk_size,
                    "overlap": self.cfg.overlap,
                }
            )]
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            # Extract chunk
            end_idx = min(start_idx + self.cfg.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                source_id=source_id,
                metadata={
                    "strategy": "fixed",
                    "token_count": len(chunk_tokens),
                    "start_token": start_idx,
                    "end_token": end_idx,
                    "chunk_size": self.cfg.chunk_size,
                    "overlap": self.cfg.overlap,
                }
            ))
            
            # Move to next chunk with overlap
            start_idx += (self.cfg.chunk_size - self.cfg.overlap)
        
        return chunks
    
    def get_stats(self, chunks: list[Chunk]) -> dict[str, Any]:
        """Compute statistics for fixed chunks."""
        if not chunks:
            return {}
        
        token_counts = [c.metadata.get("token_count", 0) for c in chunks]
        char_lengths = [len(c.text) for c in chunks]
        
        return {
            "strategy": self.strategy_name,
            "total_chunks": len(chunks),
            "avg_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
            "avg_chars": sum(char_lengths) / len(char_lengths),
            "total_tokens": sum(token_counts),
            "chunk_size": self.cfg.chunk_size,
            "overlap": self.cfg.overlap,
        }
    
    @property
    def strategy_name(self) -> str:
        return "fixed"
