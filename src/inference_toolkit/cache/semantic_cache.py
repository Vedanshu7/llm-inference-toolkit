import hashlib
import logging
import time
from dataclasses import dataclass

import litellm
import numpy as np

import inference_toolkit.cache.store as cache_store
from inference_toolkit.config import settings

_LOG = logging.getLogger(__name__)


@dataclass
class CacheHit:
    """
    Wrap a cache entry with the metadata explaining why it was returned.
    """

    entry: cache_store.CacheEntry
    matched_prompt: str
    similarity_score: float
    cache_age_seconds: float

    @property
    def response(self) -> str:
        """
        Return the cached response text for convenience.
        """
        return self.entry.response


@dataclass
class CacheStats:
    """
    Summarise cache performance metrics for a given session.
    """

    total_requests: int
    cache_hits: int
    hit_rate: float
    total_entries: int


class SemanticCache:
    """
    Cache LLM responses and serve hits for semantically similar prompts.

    Use embedding-based cosine similarity to match incoming prompts against
    stored ones, returning a `CacheHit` when the similarity score exceeds
    the configured threshold.
    """

    def __init__(self, store: cache_store.CacheStore) -> None:
        self._store = store
        self._total_requests = 0
        self._cache_hits = 0

    async def get(self, prompt: str) -> CacheHit | None:
        """
        Look up a semantically similar prompt in the cache.

        :param prompt: the user prompt to look up
        :return: CacheHit with match metadata if similarity exceeds threshold, else None
        """
        self._total_requests += 1
        entries = await self._store.get_all()
        if not entries:
            return None
        # Embed the incoming prompt and find the closest stored entry.
        query_embedding = await self._embed(prompt)
        best_entry, best_score = self._find_best_match(query_embedding, entries)
        if best_score >= settings.cache_similarity_threshold:
            self._cache_hits += 1
            key = self._make_key(best_entry.prompt)
            await self._store.increment_hits(key)
            age = time.time() - best_entry.created_at
            _LOG.debug("Cache hit (score=%.4f) for prompt: %s", best_score, prompt[:60])
            return CacheHit(
                entry=best_entry,
                matched_prompt=best_entry.prompt,
                similarity_score=round(best_score, 6),
                cache_age_seconds=round(age, 2),
            )
        _LOG.debug("Cache miss (best_score=%.4f) for prompt: %s", best_score, prompt[:60])
        return None

    async def set(
        self,
        prompt: str,
        response: str,
        model: str = "",
        cost_usd: float = 0.0,
    ) -> None:
        """
        Embed a prompt and persist it alongside its response and cost metadata.

        :param prompt: the user prompt that was answered
        :param response: the assistant response to cache
        :param model: the LLM model that produced the response
        :param cost_usd: estimated cost of the LLM call in USD
        """
        embedding = await self._embed(prompt)
        entry = cache_store.CacheEntry(
            prompt=prompt,
            response=response,
            embedding=embedding,
            model=model,
            cost_usd=cost_usd,
        )
        await self._store.set(
            key=self._make_key(prompt),
            entry=entry,
            ttl=settings.cache_ttl_seconds,
        )
        _LOG.debug("Cached response for prompt: %s", prompt[:60])

    async def stats(self) -> CacheStats:
        """
        Return current cache performance metrics.

        :return: snapshot of hit rate and entry count
        """
        total_entries = await self._store.count()
        hit_rate = self._cache_hits / self._total_requests if self._total_requests else 0.0
        return CacheStats(
            total_requests=self._total_requests,
            cache_hits=self._cache_hits,
            hit_rate=round(hit_rate, 4),
            total_entries=total_entries,
        )

    async def clear(self) -> None:
        """
        Evict all entries from the cache and reset in-session counters.
        """
        await self._store.clear()
        self._total_requests = 0
        self._cache_hits = 0
        _LOG.info("Cache cleared.")

    async def _embed(self, text: str) -> list[float]:
        """
        Generate an embedding vector for the given text.

        Uses fastembed for models prefixed with ``fastembed/`` (local, no API key
        required), and litellm for everything else.

        :param text: text to embed
        :return: embedding as a list of floats
        """
        if settings.embedding_model.startswith("fastembed/"):
            return await self._embed_fastembed(text)
        response = await litellm.aembedding(
            model=settings.embedding_model,
            input=[text],
        )
        return list(response.data[0]["embedding"])

    async def _embed_fastembed(self, text: str) -> list[float]:
        """
        Generate an embedding using the local fastembed library.

        :param text: text to embed
        :return: embedding as a list of floats
        """
        import asyncio

        from fastembed import TextEmbedding

        model_name = settings.embedding_model[len("fastembed/") :]
        if not hasattr(self, "_fastembed_model"):
            self._fastembed_model = TextEmbedding(model_name)
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self._fastembed_model.embed([text]))
        )
        return embeddings[0].tolist()

    @staticmethod
    def _find_best_match(
        query: list[float],
        entries: list[cache_store.CacheEntry],
    ) -> tuple[cache_store.CacheEntry, float]:
        """
        Find the entry whose embedding is most similar to the query vector.

        :param query: embedding of the incoming prompt
        :param entries: stored cache entries to search
        :return: (best matching entry, cosine similarity score)
        """
        assert entries, "entries must be non-empty."
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
        """
        Derive a short deterministic cache key from a prompt string.

        :param prompt: the prompt to hash
        :return: 16-character hex key
        """
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
