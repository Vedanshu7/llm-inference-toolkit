"""
Demo: Context Compression Engine

Simulates a long conversation that exceeds the context window and shows
the compressor kicking in to summarise old turns.

Run: uv run python examples/demo_compression.py
"""

import asyncio

import litellm
from dotenv import load_dotenv

load_dotenv()

from inference_toolkit.compression.compressor import ContextCompressor
from inference_toolkit.config import settings

# Use a small context model so compression triggers quickly in the demo
MODEL = "claude-haiku-4-5-20251001"
# Force a tiny context window so the compressor kicks in during the demo
DEMO_CONTEXT_LIMIT = 500  # tokens — artificially low to trigger compression

CONVERSATION_TURNS = [
    "My name is Alex and I'm building a FastAPI app.",
    "I'm using PostgreSQL as my database.",
    "I need to implement user authentication with JWT tokens.",
    "The app also needs rate limiting on the API endpoints.",
    "I'm deploying to AWS using ECS Fargate.",
    "I use GitHub Actions for CI/CD.",
    "I need to support multi-tenancy with separate schemas per tenant.",
    "The frontend is built with Next.js and uses React Query.",
    "I'm using Redis for session storage and caching.",
    "Security is important — I need input validation with Pydantic.",
    "I also need structured logging with correlation IDs.",
    "What tech stack have I described so far?",  # Should reference earlier messages
]


async def main() -> None:
    compressor = ContextCompressor()
    messages = [{"role": "system", "content": "You are a helpful senior software engineer."}]

    print("=" * 60)
    print("Context Compression Demo")
    print(f"Model: {MODEL} | Compression threshold: {settings.compression_threshold:.0%}")
    print("=" * 60)

    for i, user_input in enumerate(CONVERSATION_TURNS, 1):
        messages.append({"role": "user", "content": user_input})

        token_count_before = litellm.token_counter(model=MODEL, messages=messages)

        # Compress if needed — patch context limit so compression triggers in demo
        import unittest.mock as mock
        with mock.patch.object(compressor, "_get_context_limit", return_value=DEMO_CONTEXT_LIMIT):
            messages = await compressor.compress(messages, MODEL)

        token_count_after = litellm.token_counter(model=MODEL, messages=messages)

        compressed = token_count_after < token_count_before
        flag = " ← COMPRESSED" if compressed else ""

        response = await litellm.acompletion(
            model=MODEL,
            messages=messages,
            max_tokens=120,
        )
        assistant_reply = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": assistant_reply})

        print(f"\n[Turn {i:02d}] Tokens: {token_count_before} → {token_count_after}{flag}")
        print(f"  User : {user_input}")
        print(f"  Bot  : {assistant_reply[:120]}")

    print("\n" + "=" * 60)
    print(f"Final message count : {len(messages)}")
    print(f"Final token count   : {litellm.token_counter(model=MODEL, messages=messages)}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
