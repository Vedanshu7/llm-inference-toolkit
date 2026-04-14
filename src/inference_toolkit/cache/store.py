import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable

import redis.asyncio as aioredis

_LOG = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """
    Represent a single cached prompt-response pair with its embedding vector.
    """

    prompt: str
    response: str
    embedding: list[float]
    hits: int = 0
    created_at: float = field(default_factory=time.time)
    model: str = ""
    cost_usd: float = 0.0


@runtime_checkable
class CacheStore(Protocol):
    """
    Define the interface for cache storage backends.
    """

    async def get_all(self) -> list[CacheEntry]: ...
    async def set(self, key: str, entry: CacheEntry, ttl: int) -> None: ...
    async def increment_hits(self, key: str) -> None: ...
    async def clear(self) -> None: ...
    async def count(self) -> int: ...


class InMemoryStore:
    """
    Store cache entries in process memory with TTL-based expiry.
    """

    def __init__(self) -> None:
        # Map of cache key → (entry, expiry timestamp).
        self._data: dict[str, tuple[CacheEntry, float]] = {}

    async def get_all(self) -> list[CacheEntry]:
        """
        Return all non-expired entries, pruning stale ones in place.

        :return: list of live cache entries
        """
        now = time.time()
        # Prune expired entries before returning.
        expired_keys = [k for k, (_, exp) in self._data.items() if exp < now]
        for key in expired_keys:
            del self._data[key]
        return [entry for entry, _ in self._data.values()]

    async def set(self, key: str, entry: CacheEntry, ttl: int) -> None:
        """
        Store an entry under the given key with a TTL expiry.

        :param key: unique cache key (typically a prompt hash)
        :param entry: the cache entry to store
        :param ttl: time-to-live in seconds
        """
        self._data[key] = (entry, time.time() + ttl)

    async def increment_hits(self, key: str) -> None:
        """
        Increment the hit counter for the entry at the given key.

        :param key: cache key of the entry to update
        """
        if key in self._data:
            entry, expiry = self._data[key]
            entry.hits += 1
            self._data[key] = (entry, expiry)

    async def clear(self) -> None:
        """
        Remove all entries from the store.
        """
        self._data.clear()

    async def count(self) -> int:
        """
        Return the number of live (non-expired) entries.

        :return: count of live entries
        """
        return len(await self.get_all())


class RedisStore:
    """
    Store cache entries in Redis with native TTL support.
    """

    _PREFIX = "llm_cache:"

    def __init__(self, client: aioredis.Redis) -> None:
        self._client = client

    async def get_all(self) -> list[CacheEntry]:
        """
        Return all entries currently stored in Redis.

        :return: list of cache entries
        """
        keys = await self._client.keys(f"{self._PREFIX}*")
        if not keys:
            return []
        # Fetch all entries in a single round-trip.
        raw_entries = await self._client.mget(keys)
        entries = []
        for raw in raw_entries:
            if raw:
                data = json.loads(raw)
                entries.append(CacheEntry(**data))
        return entries

    async def set(self, key: str, entry: CacheEntry, ttl: int) -> None:
        """
        Persist an entry in Redis with an expiry.

        :param key: unique cache key
        :param entry: the cache entry to store
        :param ttl: time-to-live in seconds
        """
        await self._client.setex(
            f"{self._PREFIX}{key}",
            ttl,
            json.dumps(asdict(entry)),
        )

    async def increment_hits(self, key: str) -> None:
        """
        Increment the hit counter for the entry at the given key.

        :param key: cache key of the entry to update
        """
        full_key = f"{self._PREFIX}{key}"
        raw = await self._client.get(full_key)
        if raw:
            # Re-serialise with updated hit count, preserving remaining TTL.
            data = json.loads(raw)
            data["hits"] = data.get("hits", 0) + 1
            ttl = await self._client.ttl(full_key)
            await self._client.setex(full_key, max(ttl, 1), json.dumps(data))

    async def clear(self) -> None:
        """
        Delete all cache entries from Redis.
        """
        keys = await self._client.keys(f"{self._PREFIX}*")
        if keys:
            await self._client.delete(*keys)

    async def count(self) -> int:
        """
        Return the number of entries currently in Redis.

        :return: count of stored entries
        """
        return len(await self._client.keys(f"{self._PREFIX}*"))


async def get_store(redis_url: str) -> CacheStore:
    """
    Instantiate the appropriate cache backend.

    Return a `RedisStore` if a URL is provided, otherwise fall back to
    `InMemoryStore`.

    :param redis_url: Redis connection URL, or empty string for in-memory
    :return: an initialised cache store
    """
    if redis_url:
        client = aioredis.from_url(redis_url, decode_responses=True)
        _LOG.info("Using Redis cache backend at '%s'.", redis_url)
        return RedisStore(client)
    _LOG.info("No REDIS_URL set — using in-memory cache backend.")
    return InMemoryStore()
