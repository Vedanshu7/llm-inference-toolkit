import time

import pytest

from inference_toolkit.cache.semantic_cache import SemanticCache  # noqa: F401
from inference_toolkit.cache.store import CacheEntry, InMemoryStore
from tests.conftest import make_embedding


@pytest.mark.asyncio
async def test_cache_miss_on_empty_store(cache, mock_embed):
    result = await cache.get("What is the capital of France?")
    assert result is None


@pytest.mark.asyncio
async def test_cache_hit_on_similar_prompt(cache, monkeypatch):
    stored_embedding = make_embedding(0.9)
    query_embedding = make_embedding(0.901)  # very similar

    call_count = 0

    async def _fake_embed(model, input, **kwargs):  # noqa: A002
        nonlocal call_count
        from unittest.mock import MagicMock
        result = MagicMock()
        result.data = [{"embedding": stored_embedding if call_count == 0 else query_embedding}]
        call_count += 1
        return result

    monkeypatch.setattr("litellm.aembedding", _fake_embed)

    await cache.set("What is the capital of France?", "Paris")
    hit = await cache.get("Capital of France?")
    assert hit is not None
    assert hit.response == "Paris"


@pytest.mark.asyncio
async def test_cache_miss_on_dissimilar_prompt(cache, monkeypatch):
    async def _embed_factory(embeddings):
        idx = 0

        async def _fake(model, input, **kwargs):  # noqa: A002
            nonlocal idx
            from unittest.mock import MagicMock
            result = MagicMock()
            result.data = [{"embedding": embeddings[idx]}]
            idx = min(idx + 1, len(embeddings) - 1)
            return result

        return _fake

    similar = make_embedding(0.9)
    dissimilar = make_embedding(-0.9)  # opposite direction → cosine ≈ -1
    monkeypatch.setattr("litellm.aembedding", await _embed_factory([similar, dissimilar]))

    await cache.set("What is the capital of France?", "Paris")
    hit = await cache.get("How do I reverse a list in Python?")
    assert hit is None


@pytest.mark.asyncio
async def test_cache_stats_increments(cache, mock_embed):
    await cache.get("anything")
    await cache.get("something else")
    stats = await cache.stats()
    assert stats.total_requests == 2
    assert stats.cache_hits == 0
    assert stats.hit_rate == 0.0


@pytest.mark.asyncio
async def test_cache_clear(cache, mock_embed):
    await cache.set("some prompt", "some response")
    stats_before = await cache.stats()
    assert stats_before.total_entries == 1

    await cache.clear()
    stats_after = await cache.stats()
    assert stats_after.total_entries == 0


@pytest.mark.asyncio
async def test_in_memory_store_ttl_expiry():
    store = InMemoryStore()
    entry = CacheEntry(
        prompt="test",
        response="ok",
        embedding=make_embedding(0.5),
    )
    # Set with 1-second TTL
    await store.set("key1", entry, ttl=1)
    assert await store.count() == 1

    # Manually expire it by backdating
    store._data["key1"] = (store._data["key1"][0], time.time() - 1)
    assert await store.count() == 0
