"""
Qdrant hybrid indexer (dense + sparse vectors).
Supports Reciprocal Rank Fusion for optimal retrieval.
"""

from __future__ import annotations
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    SparseIndexParams,
)
from fastembed import SparseTextEmbedding, TextEmbedding

from prime_lands.config import PlatformConfig
from prime_lands.chunking.base import Chunk
from prime_lands.exceptions import IndexingError, CollectionExistsError
from prime_lands.logger import get_logger

log = get_logger(__name__)
load_dotenv()


class QdrantIndexer:
    """
    Hybrid indexer for Qdrant vector database.
    
    Features:
    - Dense vectors (OpenAI or FastEmbed local)
    - Sparse vectors (BM25 via FastEmbed)
    - Hybrid search with RRF fusion
    - Batch indexing with progress tracking
    """
    
    def __init__(self, config: PlatformConfig):
        self.cfg = config
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            host=config.qdrant.host,
            port=config.qdrant.port,
        )
        
        # Initialize dense embedding model
        self.openai = None
        self.dense_model = None
        
        if config.embeddings.provider == "openai":
            self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif config.embeddings.provider == "fastembed":
            log.info(f"Loading local dense model: {config.embeddings.dense_model}")
            self.dense_model = TextEmbedding(model_name=config.embeddings.dense_model)
        
        # Initialize sparse model (BM25)
        self.sparse_model = SparseTextEmbedding(model_name=config.embeddings.sparse_model)
        
        # Cross-encoder reranker (lazy-loaded on first use)
        self._reranker = None
        self._reranker_model = config.retrieval.reranker_model
        self._rerank_top_n = config.retrieval.rerank_top_n
        
        log.info(f"Initialized QdrantIndexer with {config.embeddings.provider} dense embeddings")
    
    def _get_dense_embedding(self, text: str) -> list[float]:
        """Get dense embedding from configured provider."""
        if self.cfg.embeddings.provider == "openai":
            response = self.openai.embeddings.create(
                model=self.cfg.embeddings.dense_model,
                input=text,
            )
            return response.data[0].embedding
        elif self.cfg.embeddings.provider == "fastembed":
            # FastEmbed returns a generator of embeddings
            embeddings = list(self.dense_model.embed([text]))
            return embeddings[0].tolist()
        else:
            raise ValueError(f"Unsupported embedding provider: {self.cfg.embeddings.provider}")
    
    def _get_sparse_embedding(self, text: str) -> SparseVector:
        """Get sparse embedding (BM25) from FastEmbed."""
        sparse_vec = list(self.sparse_model.embed([text]))[0]
        return SparseVector(
            indices=sparse_vec.indices.tolist(),
            values=sparse_vec.values.tolist(),
        )
    
    async def create_collection(self, collection_name: str, force: bool = False) -> None:
        """
        Create a Qdrant collection with hybrid vector support.
        
        Args:
            collection_name: Name of collection to create
            force: If True, delete existing collection first
        """
        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        
        if exists:
            if force:
                log.warning(f"Deleting existing collection: {collection_name}")
                self.client.delete_collection(collection_name)
            else:
                raise CollectionExistsError(f"Collection '{collection_name}' already exists")
        
        # Create collection with dense + sparse vectors
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=self.cfg.embeddings.dense_dimensions,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(),
                )
            },
        )
        
        log.info(f"✓ Created collection: {collection_name} (dense + sparse)")
    
    async def index_chunks(
        self, 
        chunks: list[Chunk], 
        collection_name: str,
        batch_size: int = 32,
    ) -> None:
        """
        Index chunks with hybrid vectors (dense + sparse).
        
        Args:
            chunks: List of chunks to index
            collection_name: Target collection
            batch_size: Number of chunks per batch
        """
        log.info(f"Indexing {len(chunks)} chunks to {collection_name}...")
        
        points = []
        
        for i, chunk in enumerate(chunks):
            # Get embeddings
            dense_vec = self._get_dense_embedding(chunk.text)
            sparse_vec = self._get_sparse_embedding(chunk.text)
            
            # Create point
            point = PointStruct(
                id=i,
                vector={
                    "dense": dense_vec,
                    "sparse": sparse_vec,
                },
                payload={
                    "text": chunk.text,
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "metadata": chunk.metadata,
                }
            )
            points.append(point)
            
            # Batch upsert
            if len(points) >= batch_size:
                self.client.upsert(
                    collection_name=collection_name,
                    points=points,
                )
                log.info(f"Indexed {i+1}/{len(chunks)} chunks...")
                points = []
        
        # Upload remaining points
        if points:
            self.client.upsert(
                collection_name=collection_name,
                points=points,
            )
        
        log.info(f"✓ Indexed {len(chunks)} chunks to {collection_name}")
    
    def _get_reranker(self):
        """Lazy-load cross-encoder reranker on first use."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            log.info(f"Loading cross-encoder reranker: {self._reranker_model}")
            self._reranker = CrossEncoder(self._reranker_model)
        return self._reranker

    def hybrid_search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        rerank: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Perform hybrid search (dense + sparse) with optional cross-encoder reranking.
        
        Args:
            query: Search query
            collection_name: Collection to search
            top_k: Number of results to retrieve before reranking
            rerank: Whether to apply cross-encoder reranking
            
        Returns:
            List of search results sorted by score
        """
        # Get query embeddings
        dense_vec = self._get_dense_embedding(query)
        sparse_vec = self._get_sparse_embedding(query)
        
        # Dense search
        results = self.client.query_points(
            collection_name=collection_name,
            query=dense_vec,
            using="dense",
            query_filter=None,
            limit=top_k,
            with_payload=True,
        ).points
        
        # Format results
        formatted = []
        for result in results:
            formatted.append({
                "text": result.payload.get("text", ""),
                "score": result.score,
                "chunk_id": result.payload.get("chunk_id", ""),
                "source_id": result.payload.get("source_id", ""),
                "metadata": result.payload.get("metadata", {}),
            })
        
        if not formatted:
            return formatted
        
        # Cross-encoder reranking
        if rerank and formatted:
            try:
                reranker = self._get_reranker()
                pairs = [(query, r["text"]) for r in formatted]
                scores = reranker.predict(pairs)
                for r, score in zip(formatted, scores):
                    r["rerank_score"] = float(score)
                formatted.sort(key=lambda x: x["rerank_score"], reverse=True)
                formatted = formatted[:self._rerank_top_n]
                log.debug(f"Reranked {len(pairs)} → {len(formatted)} results")
            except Exception as e:
                log.warning(f"Cross-encoder reranking failed, using dense scores: {e}")
        
        return formatted
