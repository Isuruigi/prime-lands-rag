"""
Prime Lands Platform — Production-grade Real Estate Intelligence.

A hybrid RAG system powered by Claude 3.5 Haiku and OpenAI embeddings.
"""

__version__ = "1.0.0"
__author__ = "Zuu Crew AI Engineer"

from prime_lands.config import load_config, PlatformConfig
from prime_lands.logger import setup_logger, get_logger

__all__ = [
    "load_config",
    "PlatformConfig",
    "setup_logger",
    "get_logger",
]
