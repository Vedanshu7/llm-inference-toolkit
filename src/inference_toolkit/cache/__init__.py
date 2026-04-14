"""
Cache package — semantic response cache backed by Redis or in-memory storage.
"""

from inference_toolkit.cache.semantic_cache import CacheStats, SemanticCache
from inference_toolkit.cache.store import CacheEntry, CacheStore, InMemoryStore, RedisStore, get_store

__all__ = [
    "CacheEntry",
    "CacheStats",
    "CacheStore",
    "InMemoryStore",
    "RedisStore",
    "SemanticCache",
    "get_store",
]
