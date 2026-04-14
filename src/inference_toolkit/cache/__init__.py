"""
Cache package — semantic response cache backed by Redis or in-memory storage.
"""

from inference_toolkit.cache.analytics import (
    CacheAnalytics,
    CacheEntryDetail,
    Cluster,
    SavingsReport,
)
from inference_toolkit.cache.semantic_cache import CacheHit, CacheStats, SemanticCache
from inference_toolkit.cache.store import (
    CacheEntry,
    CacheStore,
    InMemoryStore,
    RedisStore,
    get_store,
)

__all__ = [
    "CacheAnalytics",
    "CacheEntry",
    "CacheEntryDetail",
    "CacheHit",
    "CacheStats",
    "CacheStore",
    "Cluster",
    "InMemoryStore",
    "RedisStore",
    "SavingsReport",
    "SemanticCache",
    "get_store",
]
