import json
import time
from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable

import redis.asyncio as aioredis


@dataclass
class CacheEntry:
    prompt: str
    response: str
    embedding: list[float]
    hits: int = 0
    created_at: float = field(default_factory=time.time)


@runtime_checkable
class CacheStore(Protocol):
    async def get_all(self) -> list[CacheEntry]: ...
    async def set(self, key: str, entry: CacheEntry, ttl: int) -> None: ...
    async def increment_hits(self, key: str) -> None: ...
    async def clear(self) -> None: ...
    async def count(self) -> int: ...


class InMemoryStore:
    def __init__(self) -> None:
        self._data: dict[str, tuple[CacheEntry, float]] = {}  # key → (entry, expires_at)

    async def get_all(self) -> list[CacheEntry]:
        now = time.time()
        expired = [k for k, (_, exp) in self._data.items() if exp < now]
        for k in expired:
            del self._data[k]
        return [entry for entry, _ in self._data.values()]

    async def set(self, key: str, entry: CacheEntry, ttl: int) -> None:
        self._data[key] = (entry, time.time() + ttl)

    async def increment_hits(self, key: str) -> None:
        if key in self._data:
            entry, exp = self._data[key]
            entry.hits += 1

    async def clear(self) -> None:
        self._data.clear()

    async def count(self) -> int:
        return len(await self.get_all())


class RedisStore:
    _PREFIX = "llm_cache:"

    def __init__(self, client: aioredis.Redis) -> None:
        self._client = client

    async def get_all(self) -> list[CacheEntry]:
        keys = await self._client.keys(f"{self._PREFIX}*")
        if not keys:
            return []
        raw_entries = await self._client.mget(keys)
        entries = []
        for raw in raw_entries:
            if raw:
                data = json.loads(raw)
                entries.append(CacheEntry(**data))
        return entries

    async def set(self, key: str, entry: CacheEntry, ttl: int) -> None:
        await self._client.setex(
            f"{self._PREFIX}{key}",
            ttl,
            json.dumps(asdict(entry)),
        )

    async def increment_hits(self, key: str) -> None:
        full_key = f"{self._PREFIX}{key}"
        raw = await self._client.get(full_key)
        if raw:
            data = json.loads(raw)
            data["hits"] = data.get("hits", 0) + 1
            ttl = await self._client.ttl(full_key)
            await self._client.setex(full_key, max(ttl, 1), json.dumps(data))

    async def clear(self) -> None:
        keys = await self._client.keys(f"{self._PREFIX}*")
        if keys:
            await self._client.delete(*keys)

    async def count(self) -> int:
        return len(await self._client.keys(f"{self._PREFIX}*"))


async def get_store(redis_url: str) -> CacheStore:
    if redis_url:
        client = aioredis.from_url(redis_url, decode_responses=True)
        return RedisStore(client)
    return InMemoryStore()
