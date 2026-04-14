from unittest.mock import MagicMock

import pytest

from inference_toolkit.cache.semantic_cache import SemanticCache
from inference_toolkit.cache.store import InMemoryStore
from inference_toolkit.compression.compressor import ContextCompressor
from inference_toolkit.conversation.manager import (
    BudgetExceededError,
    Conversation,
    ConversationStore,
)


def _make_cache() -> SemanticCache:
    return SemanticCache(store=InMemoryStore())


def _make_compressor() -> ContextCompressor:
    return ContextCompressor()


def _fake_llm_response(content: str = "Test response") -> MagicMock:
    """
    Build a minimal litellm-like response mock.
    """
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage.total_tokens = 50
    response.model_dump.return_value = {}
    return response


@pytest.mark.asyncio
async def test_conversation_store_create_and_get():
    store = ConversationStore()
    conv = await store.create(model="gpt-4o-mini")
    assert conv.id
    assert conv.model == "gpt-4o-mini"
    retrieved = await store.get(conv.id)
    assert retrieved is not None
    assert retrieved.id == conv.id


@pytest.mark.asyncio
async def test_conversation_store_delete():
    store = ConversationStore()
    conv = await store.create(model="gpt-4o-mini")
    deleted = await store.delete(conv.id)
    assert deleted is True
    assert await store.get(conv.id) is None


@pytest.mark.asyncio
async def test_conversation_store_delete_nonexistent():
    store = ConversationStore()
    deleted = await store.delete("nonexistent-id")
    assert deleted is False


@pytest.mark.asyncio
async def test_send_returns_turn_on_llm_call(monkeypatch):
    cache = _make_cache()
    compressor = _make_compressor()
    # Patch embedding to always return a miss (unique vector per call).
    call_count = 0

    async def _fake_embed(model, input, **kwargs):  # noqa: A002
        nonlocal call_count
        call_count += 1
        # Each call gets an orthogonal unit vector to guarantee cache misses.
        vec = [0.0] * 8
        vec[call_count % 8] = 1.0
        result = MagicMock()
        result.data = [{"embedding": vec}]
        return result

    monkeypatch.setattr("litellm.aembedding", _fake_embed)
    monkeypatch.setattr("litellm.token_counter", lambda **_: 10)
    monkeypatch.setattr("litellm.get_max_tokens", lambda _: 8192)
    monkeypatch.setattr("litellm.acompletion", lambda **_: _make_llm_future("Hello!"))
    monkeypatch.setattr("litellm.completion_cost", lambda **_: 0.001)

    conv = Conversation(id="test-1", model="gpt-4o-mini")
    turn = await conv.send("Hi there", cache=cache, compressor=compressor)
    assert turn.response == "Hello!"
    assert turn.cache_hit is None
    assert turn.cost_usd == 0.001
    assert turn.cumulative_cost_usd == 0.001


async def _make_llm_future(content: str) -> MagicMock:
    return _fake_llm_response(content)


@pytest.mark.asyncio
async def test_cost_accumulates_across_turns(monkeypatch):
    cache = _make_cache()
    compressor = _make_compressor()
    call_count = 0

    async def _fake_embed(model, input, **kwargs):  # noqa: A002
        nonlocal call_count
        call_count += 1
        # Each call gets an orthogonal unit vector to guarantee cache misses.
        vec = [0.0] * 8
        vec[call_count % 8] = 1.0
        result = MagicMock()
        result.data = [{"embedding": vec}]
        return result

    monkeypatch.setattr("litellm.aembedding", _fake_embed)
    monkeypatch.setattr("litellm.token_counter", lambda **_: 10)
    monkeypatch.setattr("litellm.get_max_tokens", lambda _: 8192)
    monkeypatch.setattr("litellm.acompletion", lambda **_: _make_llm_future("ok"))
    monkeypatch.setattr("litellm.completion_cost", lambda **_: 0.005)

    conv = Conversation(id="test-2", model="gpt-4o-mini")
    await conv.send("Turn 1", cache=cache, compressor=compressor)
    await conv.send("Turn 2", cache=cache, compressor=compressor)
    assert abs(conv.cumulative_cost_usd - 0.01) < 1e-9


@pytest.mark.asyncio
async def test_budget_exceeded_raises(monkeypatch):
    monkeypatch.setattr(
        "inference_toolkit.conversation.manager.settings.max_cost_usd_per_conversation",
        0.001,
    )
    conv = Conversation(id="test-3", model="gpt-4o-mini", cumulative_cost_usd=0.002)
    with pytest.raises(BudgetExceededError):
        conv._check_budget()


@pytest.mark.asyncio
async def test_serialise_deserialise_roundtrip():
    conv = Conversation(
        id="abc-123",
        model="claude-3-haiku",
        messages=[{"role": "user", "content": "Hi"}],
        cumulative_cost_usd=0.042,
        created_at=1_700_000_000.0,
    )
    data = conv.serialise()
    restored = Conversation.deserialise(data)
    assert restored.id == conv.id
    assert restored.model == conv.model
    assert restored.cumulative_cost_usd == conv.cumulative_cost_usd
    assert restored.messages == conv.messages
    assert restored.created_at == conv.created_at
