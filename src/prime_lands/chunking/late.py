"""
Late chunking (contextual embedding).
Embeds full text first, then chunks - preserves global context.
Inspired by JINA AI's late chunking approach.
"""

from __future__ import annotations
import hashlib
from typing import Any

import tiktoken

from prime_lands.chunking.base import AbstractChunker, Chunk
from prime_lands.config import LateChunkConfig
from prime_lands.logger import get_logger

log = get_logger(__name__)


class LateChunker(AbstractChunker):
    """
    Late chunking (contextual embedding).
    
    Traditional: chunk text → embed each chunk independently
    Late: embed full text → extract chunk representations from full embedding
    
    Benefits:
    - Each chunk embedding has global document context
    - Better for long documents where early chunks reference later content
    
    Implementation:
    - Simulate by adding surrounding context to each chunk
    - Prepend partial context window before target chunk
    """
    
    def __init__(self, config: LateChunkConfig):
        self.cfg = config
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def chunk(self, text: str, source_id: str = "") -> list[Chunk]:
        """
        Create chunks with surrounding context prepended.
        
        Args:
            text: Input text
            source_id: Source document ID
            
        Returns:
            List of contextually-enriched chunks
        """
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= self.cfg.context_window:
            chunk_id = hashlib.md5(text.encode()).hexdigest()[:12]
            return [Chunk(
                text=text,
                chunk_id=chunk_id,
                source_id=source_id,
                metadata={
                    "strategy": "late",
                    "token_count": len(tokens),
                    "has_prefix_context": False,
                }
            )]
        
        chunks = []
        chunk_size = int(self.cfg.context_window * (1 - self.cfg.prepend_ratio))
        prefix_size = int(self.cfg.context_window * self.cfg.prepend_ratio)
        
        start_idx = 0
        
        while start_idx < len(tokens):
            # Extract main chunk
            chunk_end = min(start_idx + chunk_size, len(tokens))
            
            # Add prefix context from before this chunk
            prefix_start = max(0, start_idx - prefix_size)
            prefix_tokens = tokens[prefix_start:start_idx]
            chunk_tokens = tokens[start_idx:chunk_end]
            
            # Combine prefix + chunk
            full_tokens = prefix_tokens + chunk_tokens
            full_text = self.encoding.decode(full_tokens)
            
            # Only the main chunk text (without prefix) for pure content
            chunk_text_only = self.encoding.decode(chunk_tokens)
            
            chunk_id = hashlib.md5(chunk_text_only.encode()).hexdigest()[:12]
            chunks.append(Chunk(
                text=full_text,  # Full text includes context
                chunk_id=chunk_id,
                source_id=source_id,
                metadata={
                    "strategy": "late",
                    "token_count": len(chunk_tokens),
                    "prefix_tokens": len(prefix_tokens),
                    "total_tokens": len(full_tokens),
                    "has_prefix_context": len(prefix_tokens) > 0,
                    "chunk_text_only": chunk_text_only,  # Original chunk without context
                }
            ))
            
            start_idx += chunk_size
        
        return chunks
    
    def get_stats(self, chunks: list[Chunk]) -> dict[str, Any]:
        """Compute statistics for late chunks."""
        if not chunks:
            return {}
        
        token_counts = [c.metadata.get("token_count", 0) for c in chunks]
        total_tokens = [c.metadata.get("total_tokens", 0) for c in chunks]
        with_context = sum(1 for c in chunks if c.metadata.get("has_prefix_context", False))
        
        return {
            "strategy": self.strategy_name,
            "total_chunks": len(chunks),
            "avg_chunk_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
            "avg_total_tokens": sum(total_tokens) / len(total_tokens) if total_tokens else 0,
            "chunks_with_context": with_context,
            "context_percentage": (with_context / len(chunks) * 100) if chunks else 0,
            "prepend_ratio": self.cfg.prepend_ratio,
        }
    
    @property
    def strategy_name(self) -> str:
        return "late"
