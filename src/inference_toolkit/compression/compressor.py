import logging

import litellm

from inference_toolkit.config import settings

_LOG = logging.getLogger(__name__)

# Type alias for a single chat message dict.
Message = dict[str, str]

_SUMMARISE_SYSTEM = (
    "You are a conversation summariser. Given a list of messages, produce a concise factual "
    "summary in 3–5 sentences. Preserve key decisions, facts, code snippets, and any "
    "important context. Write in third person. Do not add commentary or opinions."
)

# Token limit used when litellm cannot resolve the model's context window.
_FALLBACK_CONTEXT_WINDOW = 8_192


class ContextCompressor:
    """
    Automatically summarise old conversation turns when approaching a model's context limit.

    When the running token count exceeds `compression_threshold × context_window`,
    compress the oldest non-system messages into a single summary turn and continue
    the conversation without truncation.
    """

    async def compress(self, messages: list[Message], model: str) -> list[Message]:
        """
        Compress messages in place if they approach the model's context limit.

        :param messages: the full conversation history
        :param model: the LLM model identifier (used for token counting and limit lookup)
        :return: the original messages unchanged, or a compressed version with a summary
        """
        assert messages, "messages must be non-empty."
        current_tokens = litellm.token_counter(model=model, messages=messages)
        context_limit = self._get_context_limit(model)
        threshold_tokens = int(context_limit * settings.compression_threshold)
        if current_tokens <= threshold_tokens:
            return messages
        _LOG.info(
            "Compressing context: %d tokens > %d threshold (model=%s).",
            current_tokens,
            threshold_tokens,
            model,
        )
        # Separate system prompt (must be preserved) from conversational turns.
        system_messages = [m for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]
        # Require at least 3 non-system messages before attempting compression.
        if len(non_system) <= 2:
            return messages
        # Keep the two most recent messages intact; summarise everything older.
        to_compress = non_system[:-2]
        to_keep = non_system[-2:]
        summary = await self._summarise(to_compress, model)
        summary_message: Message = {
            "role": "user",
            "content": f"[Summary of earlier conversation]\n{summary}",
        }
        compressed = system_messages + [summary_message] + to_keep
        new_tokens = litellm.token_counter(model=model, messages=compressed)
        _LOG.info(
            "Compressed %d → %d tokens (saved %d).",
            current_tokens,
            new_tokens,
            current_tokens - new_tokens,
        )
        return compressed

    async def _summarise(self, messages: list[Message], model: str) -> str:
        """
        Produce a concise summary of the given messages using a cheap fast model.

        :param messages: conversation turns to summarise
        :param model: original model (unused, kept for future per-model routing)
        :return: summary text
        """
        formatted = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        response = await litellm.acompletion(
            model=settings.compression_model,
            messages=[
                {"role": "system", "content": _SUMMARISE_SYSTEM},
                {"role": "user", "content": formatted},
            ],
            max_tokens=512,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _get_context_limit(model: str) -> int:
        """
        Look up the context window size for the given model.

        Fall back to `_FALLBACK_CONTEXT_WINDOW` if litellm cannot resolve it.

        :param model: LLM model identifier
        :return: max token count for the model's context window
        """
        try:
            limit = litellm.get_max_tokens(model)
            return limit if limit else _FALLBACK_CONTEXT_WINDOW
        except Exception:
            return _FALLBACK_CONTEXT_WINDOW
