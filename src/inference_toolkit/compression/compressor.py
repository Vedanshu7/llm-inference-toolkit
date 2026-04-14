import logging

import litellm

from inference_toolkit.config import settings

logger = logging.getLogger(__name__)

Message = dict[str, str]

_SUMMARISE_SYSTEM = (
    "You are a conversation summariser. Given a list of messages, produce a concise factual "
    "summary in 3–5 sentences. Preserve key decisions, facts, code snippets, and any "
    "important context. Write in third person. Do not add commentary or opinions."
)

# Fallback context window sizes if litellm cannot resolve the model
_FALLBACK_CONTEXT_WINDOW = 8_192


class ContextCompressor:
    async def compress(self, messages: list[Message], model: str) -> list[Message]:
        current_tokens = litellm.token_counter(model=model, messages=messages)
        context_limit = self._get_context_limit(model)
        threshold_tokens = int(context_limit * settings.compression_threshold)

        if current_tokens <= threshold_tokens:
            return messages

        logger.info(
            "Compressing context: %d tokens > %d threshold (model=%s)",
            current_tokens,
            threshold_tokens,
            model,
        )

        system_messages = [m for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]

        # Keep the most recent turn intact; compress everything older
        if len(non_system) <= 2:
            return messages

        to_compress = non_system[:-2]
        to_keep = non_system[-2:]

        summary = await self._summarise(to_compress, model)
        summary_message: Message = {
            "role": "user",
            "content": f"[Summary of earlier conversation]\n{summary}",
        }

        compressed = system_messages + [summary_message] + to_keep
        new_tokens = litellm.token_counter(model=model, messages=compressed)
        logger.info(
            "Compressed %d → %d tokens (saved %d)",
            current_tokens,
            new_tokens,
            current_tokens - new_tokens,
        )
        return compressed

    async def _summarise(self, messages: list[Message], model: str) -> str:
        formatted = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )
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
        try:
            limit = litellm.get_max_tokens(model)
            return limit if limit else _FALLBACK_CONTEXT_WINDOW
        except Exception:
            return _FALLBACK_CONTEXT_WINDOW
