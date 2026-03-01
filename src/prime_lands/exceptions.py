"""
Custom exception hierarchy for Prime Lands platform.
Never let raw Exception bubble up — always be specific.
"""


class PrimeLandsError(Exception):
    """Base exception for all platform errors."""
    pass


# ─── Crawler Exceptions ───────────────────────────────────────
class CrawlerError(PrimeLandsError):
    """Base class for crawler failures."""
    pass


class PageLoadError(CrawlerError):
    """Failed to load a page after all retries."""
    def __init__(self, url: str, attempts: int):
        super().__init__(f"Failed to load {url} after {attempts} attempts")
        self.url = url
        self.attempts = attempts


class ExtractionError(CrawlerError):
    """Failed to extract required fields from a page."""
    def __init__(self, url: str, field: str):
        super().__init__(f"Failed to extract '{field}' from {url}")
        self.url = url
        self.field = field


class RobotsBlocked(CrawlerError):
    """URL disallowed by robots.txt."""
    def __init__(self, url: str):
        super().__init__(f"robots.txt blocks crawling: {url}")
        self.url = url


# ─── Chunking Exceptions ──────────────────────────────────────
class ChunkingError(PrimeLandsError):
    """Base class for chunking failures."""
    pass


class EmptyDocumentError(ChunkingError):
    """Document has no content to chunk."""
    def __init__(self, doc_id: str):
        super().__init__(f"Document '{doc_id}' has no extractable text")
        self.doc_id = doc_id


# ─── Indexing Exceptions ──────────────────────────────────────
class IndexingError(PrimeLandsError):
    """Qdrant indexing failure."""
    pass


class CollectionExistsError(IndexingError):
    """Collection already exists and force=False."""
    pass


# ─── Service Exceptions ───────────────────────────────────────
class ServiceError(PrimeLandsError):
    """Base class for intelligence service failures."""
    pass


class RetrievalError(ServiceError):
    """Vector search failed."""
    pass


class GenerationError(ServiceError):
    """LLM generation failed."""
    pass


class CacheError(ServiceError):
    """Cache operation failed."""
    pass
