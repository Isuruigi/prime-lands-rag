"""
Semantic chunking using sentence similarity (spaCy).
Groups sentences until similarity drops below threshold.
"""

from __future__ import annotations
import hashlib
import re
from typing import Any

import spacy
from sentence_transformers import SentenceTransformer, util

from prime_lands.chunking.base import AbstractChunker, Chunk
from prime_lands.config import SemanticChunkConfig
from prime_lands.logger import get_logger

log = get_logger(__name__)


class SemanticChunker(AbstractChunker):
    """
    Semantic chunking via sentence embeddings + similarity thresholding.
    
    Algorithm:
    1. Split text into sentences (spaCy/NLTK/regex)
    2. Embed each sentence
    3. Group consecutive sentences while cosine similarity > threshold
    4. Enforce min/max token limits
    
    This preserves semantic coherence better than fixed-size chunking.
    """
    
    def __init__(self, config: SemanticChunkConfig):
        self.cfg = config
        self.model = SentenceTransformer(config.model)
        
        # Load sentence splitter
        if config.sentence_splitter == "spacy":
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                log.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
                log.info("Falling back to regex sentence splitter")
                self.nlp = None
        else:
            self.nlp = None
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        else:
            # Fallback regex sentence splitter
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def chunk(self, text: str, source_id: str = "") -> list[Chunk]:
        """
        Chunk text using semantic similarity between sentences.
        
        Args:
            text: Input text
            source_id: Source document ID
            
        Returns:
            List of semantically coherent chunks
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        
        # Embed all sentences
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # Compute similarity between current sentence and previous
            sim = util.cos_sim(embeddings[i-1], embeddings[i]).item()
            
            # Estimate token count (rough: 1 word ≈ 1.3 tokens)
            current_text = " ".join(current_chunk)
            estimated_tokens = len(current_text.split()) * 1.3
            
            # Decision: add to current chunk or start new one
            if (sim >= self.cfg.similarity_threshold and 
                estimated_tokens < self.cfg.max_tokens):
                current_chunk.append(sentences[i])
            else:
                # Finalize current chunk
                chunk_text = " ".join(current_chunk)
                if len(chunk_text.split()) * 1.3 >= self.cfg.min_tokens:
                    chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
                    chunks.append(Chunk(
                        text=chunk_text,
                        chunk_id=chunk_id,
                        source_id=source_id,
                        metadata={
                            "strategy": "semantic",
                            "sentence_count": len(current_chunk),
                            "similarity_threshold": self.cfg.similarity_threshold,
                        }
                    ))
                
                # Start new chunk
                current_chunk = [sentences[i]]
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                source_id=source_id,
                metadata={
                    "strategy": "semantic",
                    "sentence_count": len(current_chunk),
                    "similarity_threshold": self.cfg.similarity_threshold,
                }
            ))
        
        return chunks
    
    def get_stats(self, chunks: list[Chunk]) -> dict[str, Any]:
        """Compute statistics for semantic chunks."""
        if not chunks:
            return {}
        
        lengths = [len(c.text) for c in chunks]
        word_counts = [len(c.text.split()) for c in chunks]
        
        return {
            "strategy": self.strategy_name,
            "total_chunks": len(chunks),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "avg_words": sum(word_counts) / len(word_counts),
            "total_chars": sum(lengths),
        }
    
    @property
    def strategy_name(self) -> str:
        return "semantic"
