from unittest.mock import MagicMock

import pytest

from inference_toolkit.compression.compressor import ContextCompressor  # noqa: F401


def _make_messages(n_turns: int, include_system: bool = True) -> list[dict]:
    messages = []
    if include_system:
        messages.append({"role": "system", "content": "You are a helpful assistant."})
    for i in range(n_turns):
        messages.append({"role": "user", "content": f"User message {i}"})
        messages.append({"role": "assistant", "content": f"Assistant response {i}"})
    return messages


@pytest.mark.asyncio
async def test_no_compression_below_threshold(compressor, monkeypatch):
    monkeypatch.setattr("litellm.token_counter", lambda **_: 100)
    monkeypatch.setattr("litellm.get_max_tokens", lambda _: 8192)

    messages = _make_messages(3)
    result = await compressor.compress(messages, "gpt-4o")
    assert result == messages


@pytest.mark.asyncio
async def test_compression_triggers_above_threshold(compressor, monkeypatch):
    monkeypatch.setattr("litellm.token_counter", lambda **_: 7000)
    monkeypatch.setattr("litellm.get_max_tokens", lambda _: 8192)

    async def _fake_summary(*args, **kwargs):
        result = MagicMock()
        result.choices = [MagicMock()]
        result.choices[0].message.content = "Summary of earlier conversation."
        return result

    monkeypatch.setattr("litellm.acompletion", _fake_summary)

    messages = _make_messages(10)
    result = await compressor.compress(messages, "gpt-4o")

    # System message must be preserved
    assert result[0]["role"] == "system"
    # Result should be shorter than input
    assert len(result) < len(messages)
    # A summary message should be present
    assert any("Summary" in m["content"] for m in result)


@pytest.mark.asyncio
async def test_system_message_always_preserved(compressor, monkeypatch):
    monkeypatch.setattr("litellm.token_counter", lambda **_: 9000)
    monkeypatch.setattr("litellm.get_max_tokens", lambda _: 8192)

    async def _fake_summary(*args, **kwargs):
        result = MagicMock()
        result.choices = [MagicMock()]
        result.choices[0].message.content = "Summary."
        return result

    monkeypatch.setattr("litellm.acompletion", _fake_summary)

    messages = _make_messages(15, include_system=True)
    result = await compressor.compress(messages, "gpt-4o")

    system_messages = [m for m in result if m["role"] == "system"]
    assert len(system_messages) == 1
    assert system_messages[0]["content"] == "You are a helpful assistant."


@pytest.mark.asyncio
async def test_no_compression_with_too_few_messages(compressor, monkeypatch):
    monkeypatch.setattr("litellm.token_counter", lambda **_: 9000)
    monkeypatch.setattr("litellm.get_max_tokens", lambda _: 8192)

    # Only 2 non-system messages — compressor should leave them alone
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    result = await compressor.compress(messages, "gpt-4o")
    assert result == messages
