"""
Demo: Semantic Response Cache

Fires a set of semantically similar questions and shows cache hits + savings.
Run: uv run python examples/demo_cache.py
"""

import asyncio
import time

import litellm
from dotenv import load_dotenv

load_dotenv()

from inference_toolkit.cache.semantic_cache import SemanticCache
from inference_toolkit.cache.store import get_store
from inference_toolkit.config import settings

QUESTIONS = [
    "What is the capital of France?",          # original — will be cached
    "What's the capital city of France?",       # similar — should hit cache
    "Can you tell me France's capital?",        # similar — should hit cache
    "How do I reverse a list in Python?",       # different — cache miss
    "What is the main city of France?",         # similar — should hit cache
]

MODEL = "gpt-4o-mini"


async def main() -> None:
    store = await get_store(settings.redis_url)
    cache = SemanticCache(store)

    print("=" * 60)
    print("Semantic Cache Demo")
    print("=" * 60)

    total_api_calls = 0
    total_time = 0.0

    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n[{i}] {question}")
        start = time.perf_counter()

        cached = await cache.get(question)
        if cached:
            elapsed = time.perf_counter() - start
            print(f"    HIT  ({elapsed*1000:.1f}ms) → {cached.response[:80]}")
        else:
            response = await litellm.acompletion(
                model=MODEL,
                messages=[{"role": "user", "content": question}],
                max_tokens=60,
            )
            answer = response.choices[0].message.content.strip()
            await cache.set(question, answer)
            elapsed = time.perf_counter() - start
            total_api_calls += 1
            print(f"    MISS ({elapsed*1000:.1f}ms) → {answer[:80]}")

        total_time += elapsed

    stats = await cache.stats()
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Total questions : {stats.total_requests}")
    print(f"  Cache hits      : {stats.cache_hits}")
    print(f"  Hit rate        : {stats.hit_rate:.0%}")
    print(f"  API calls made  : {total_api_calls}")
    print(f"  API calls saved : {stats.cache_hits}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
