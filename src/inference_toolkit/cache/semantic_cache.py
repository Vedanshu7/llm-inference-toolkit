import hashlib
import logging
from dataclasses import dataclass

import litellm
import numpy as np

from inference_toolkit.cache.store import CacheEntry, CacheStore
from inference_toolkit.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    total_requests: int
    cache_hits: int
    hit_rate: float
    total_entries: int


class SemanticCache:
    def __init__(self, store: CacheStore) -> None:
        self._store = store
        self._total_requests = 0
        self._cache_hits = 0

    async def get(self, prompt: str) -> CacheEntry | None:
        self._total_requests += 1
        entries = await self._store.get_all()
        if not entries:
            return None

        query_embedding = await self._embed(prompt)
        best_entry, best_score = self._find_best_match(query_embedding, entries)

        if best_score >= settings.cache_similarity_threshold:
            self._cache_hits += 1
            key = self._make_key(best_entry.prompt)
            await self._store.increment_hits(key)
            logger.debug("Cache hit (score=%.4f) for prompt: %s", best_score, prompt[:60])
            return best_entry

        logger.debug("Cache miss (best_score=%.4f) for prompt: %s", best_score, prompt[:60])
        return None

    async def set(self, prompt: str, response: str) -> None:
        embedding = await self._embed(prompt)
        entry = CacheEntry(prompt=prompt, response=response, embedding=embedding)
        await self._store.set(
            key=self._make_key(prompt),
            entry=entry,
            ttl=settings.cache_ttl_seconds,
        )
        logger.debug("Cached response for prompt: %s", prompt[:60])

    async def stats(self) -> CacheStats:
        total_entries = await self._store.count()
        hit_rate = self._cache_hits / self._total_requests if self._total_requests else 0.0
        return CacheStats(
            total_requests=self._total_requests,
            cache_hits=self._cache_hits,
            hit_rate=round(hit_rate, 4),
            total_entries=total_entries,
        )

    async def clear(self) -> None:
        await self._store.clear()
        self._total_requests = 0
        self._cache_hits = 0
        logger.info("Cache cleared")

    async def _embed(self, text: str) -> list[float]:
        response = await litellm.aembedding(
            model=settings.embedding_model,
            input=[text],
        )
        return response.data[0]["embedding"]

    @staticmethod
    def _find_best_match(
        query: list[float],
        entries: list[CacheEntry],
    ) -> tuple[CacheEntry, float]:
        q = np.array(query)
        best_entry = entries[0]
        best_score = -1.0

        for entry in entries:
            v = np.array(entry.embedding)
            score = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-10))
            if score > best_score:
                best_score = score
                best_entry = entry

        return best_entry, best_score

    @staticmethod
    def _make_key(prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
