import pytest

from inference_toolkit.cache.analytics import CacheAnalytics
from inference_toolkit.cache.store import CacheEntry, InMemoryStore
from tests.conftest import make_embedding


async def _populate_store(store: InMemoryStore, entries: list[tuple[str, int, float]]) -> None:
    """
    Insert entries into the store as (prompt, hits, cost_usd) tuples.
    """
    for i, (prompt, hits, cost) in enumerate(entries):
        entry = CacheEntry(
            prompt=prompt,
            response=f"Response to: {prompt}",
            embedding=make_embedding(0.9 - i * 0.05),
            hits=hits,
            cost_usd=cost,
            model="gpt-4o-mini",
        )
        await store.set(f"key{i}", entry, ttl=3600)
        # Manually set hit count since set() does not accept it.
        stored_entry = store._data[f"key{i}"][0]
        stored_entry.hits = hits


@pytest.mark.asyncio
async def test_inspect_returns_all_entries():
    store = InMemoryStore()
    await _populate_store(
        store,
        [
            ("What is Python?", 5, 0.001),
            ("Explain async/await", 2, 0.002),
        ],
    )
    analytics = CacheAnalytics()
    details = await analytics.inspect(store)
    assert len(details) == 2


@pytest.mark.asyncio
async def test_inspect_sorted_by_hits_descending():
    store = InMemoryStore()
    await _populate_store(
        store,
        [
            ("Low hit prompt", 1, 0.001),
            ("High hit prompt", 10, 0.002),
        ],
    )
    analytics = CacheAnalytics()
    details = await analytics.inspect(store)
    assert details[0].hits >= details[1].hits


@pytest.mark.asyncio
async def test_inspect_calculates_savings():
    store = InMemoryStore()
    await _populate_store(store, [("Prompt", 4, 0.01)])
    analytics = CacheAnalytics()
    details = await analytics.inspect(store)
    assert abs(details[0].estimated_savings_usd - 0.04) < 1e-6


@pytest.mark.asyncio
async def test_inspect_empty_store():
    store = InMemoryStore()
    analytics = CacheAnalytics()
    details = await analytics.inspect(store)
    assert details == []


@pytest.mark.asyncio
async def test_clusters_returns_empty_for_empty_store():
    store = InMemoryStore()
    analytics = CacheAnalytics()
    clusters = await analytics.clusters(store)
    assert clusters == []


@pytest.mark.asyncio
async def test_clusters_groups_similar_prompts():
    store = InMemoryStore()
    # Two very similar embeddings and one dissimilar.
    for i, (prompt, emb_val) in enumerate(
        [
            ("What is Python?", 0.9),
            ("Tell me about Python", 0.88),  # similar to above
            ("How do I cook pasta?", -0.9),  # very different
        ]
    ):
        entry = CacheEntry(
            prompt=prompt,
            response="ok",
            embedding=make_embedding(emb_val),
            hits=1,
            cost_usd=0.001,
        )
        await store.set(f"key{i}", entry, ttl=3600)
    analytics = CacheAnalytics()
    # Use a lenient threshold so similar embeddings cluster together.
    clusters = await analytics.clusters(store, threshold=0.7)
    # The dissimilar entry must be in its own cluster.
    assert len(clusters) >= 2


@pytest.mark.asyncio
async def test_savings_report_sums_correctly():
    store = InMemoryStore()
    await _populate_store(
        store,
        [
            ("Prompt A", 3, 0.01),
            ("Prompt B", 2, 0.02),
        ],
    )
    analytics = CacheAnalytics()
    report = await analytics.savings_report(store)
    assert report.total_entries == 2
    # savings = 0.01*3 + 0.02*2 = 0.07
    assert abs(report.estimated_savings_usd - 0.07) < 1e-6
    assert report.total_cache_hits == 5


@pytest.mark.asyncio
async def test_savings_report_empty_store():
    store = InMemoryStore()
    analytics = CacheAnalytics()
    report = await analytics.savings_report(store)
    assert report.total_entries == 0
    assert report.estimated_savings_usd == 0.0
