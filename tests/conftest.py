from unittest.mock import AsyncMock, MagicMock

import pytest

from inference_toolkit.cache.store import CacheEntry, InMemoryStore
from inference_toolkit.cache.semantic_cache import SemanticCache
from inference_toolkit.compression.compressor import ContextCompressor


def make_embedding(value: float, size: int = 8) -> list[float]:
    """Create a normalised embedding pointing in a given direction."""
    import numpy as np
    v = np.zeros(size)
    v[0] = value
    v[1] = 1.0 - abs(value)
    norm = np.linalg.norm(v)
    return (v / norm).tolist()


@pytest.fixture
def in_memory_store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def cache(in_memory_store: InMemoryStore) -> SemanticCache:
    return SemanticCache(store=in_memory_store)


@pytest.fixture
def compressor() -> ContextCompressor:
    return ContextCompressor()


@pytest.fixture
def mock_embed(monkeypatch):
    """Patch litellm.aembedding to return deterministic embeddings."""
    async def _fake_embed(model, input, **kwargs):  # noqa: A002
        result = MagicMock()
        result.data = [{"embedding": make_embedding(0.9)}]
        return result

    monkeypatch.setattr("litellm.aembedding", _fake_embed)
    return _fake_embed


@pytest.fixture
def mock_completion(monkeypatch):
    """Patch litellm.acompletion to return a fixed assistant message."""
    async def _fake_completion(model, messages, **kwargs):
        result = MagicMock()
        result.choices = [MagicMock()]
        result.choices[0].message.content = "Mocked LLM response"
        result.model_dump.return_value = {
            "id": "mock",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Mocked LLM response"},
                    "finish_reason": "stop",
                }
            ],
        }
        return result

    monkeypatch.setattr("litellm.acompletion", _fake_completion)
    return _fake_completion
