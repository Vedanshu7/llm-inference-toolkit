"""
Demo: Interactive Chatbot with Semantic Cache + Context Compression

A REPL chatbot that shows both features working together.
Type 'stats' to see cache stats, 'quit' to exit.

Run: uv run python examples/demo_chatbot.py
"""

import asyncio

import litellm
from dotenv import load_dotenv

load_dotenv()

from inference_toolkit.cache.semantic_cache import SemanticCache
from inference_toolkit.cache.store import get_store
from inference_toolkit.compression.compressor import ContextCompressor
from inference_toolkit.config import settings

MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = "You are a helpful assistant. Be concise."


async def main() -> None:
    store = await get_store(settings.redis_url)
    cache = SemanticCache(store)
    compressor = ContextCompressor()

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("=" * 60)
    print("LLM Chatbot — Semantic Cache + Context Compression")
    print(f"Model: {MODEL}")
    print("Commands: 'stats' | 'clear' | 'quit'")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye.")
            break

        if user_input.lower() == "stats":
            stats = await cache.stats()
            print(f"\n  Requests  : {stats.total_requests}")
            print(f"  Hits      : {stats.cache_hits}")
            print(f"  Hit rate  : {stats.hit_rate:.0%}")
            print(f"  Entries   : {stats.total_entries}")
            print(f"  Messages  : {len(messages)}")
            print(f"  Tokens    : {litellm.token_counter(model=MODEL, messages=messages)}")
            continue

        if user_input.lower() == "clear":
            await cache.clear()
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("  Cache and history cleared.")
            continue

        # Check cache first
        cached = await cache.get(user_input)
        if cached:
            print(f"Bot (cached): {cached.response}")
            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": cached.response})
            continue

        # Compress context if needed
        messages.append({"role": "user", "content": user_input})
        messages_before = len(messages)
        messages = await compressor.compress(messages, MODEL)
        if len(messages) < messages_before:
            print("  [context compressed]")

        # Call LLM
        response = await litellm.acompletion(model=MODEL, messages=messages, max_tokens=256)
        reply = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": reply})

        # Cache the response
        await cache.set(user_input, reply)

        print(f"Bot: {reply}")


if __name__ == "__main__":
    asyncio.run(main())
