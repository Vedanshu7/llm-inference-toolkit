# LLM Inference Toolkit

A production-quality middleware layer for LLM inference with two core features:

- **Semantic Response Cache** — returns cached responses for semantically similar prompts, cutting API costs without requiring exact matches
- **Context Compression Engine** — automatically summarises old conversation turns when approaching a model's context limit

Supports any LLM provider via [litellm](https://github.com/BerriAI/litellm) (Anthropic, OpenAI, Gemini, and 100+ others). Exposed as an OpenAI-compatible REST API.

---

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Run locally (in-memory cache)

```bash
uv run uvicorn inference_toolkit.api.main:app --reload
```

### 4. Run with Docker + Redis

```bash
docker-compose up
```

---

## API

The service exposes an OpenAI-compatible API, so it works as a drop-in proxy.

### `POST /v1/chat/completions`

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

Send the same (or similar) question again — the response will be returned from cache instantly.

### `GET /v1/cache/stats`

```bash
curl http://localhost:8000/v1/cache/stats
```

```json
{
  "total_requests": 10,
  "cache_hits": 7,
  "hit_rate": 0.7,
  "total_entries": 3
}
```

### `DELETE /v1/cache`

```bash
curl -X DELETE http://localhost:8000/v1/cache
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `REDIS_URL` | *(empty)* | Redis URL — falls back to in-memory if not set |
| `CACHE_SIMILARITY_THRESHOLD` | `0.92` | Cosine similarity threshold for cache hits (0–1) |
| `CACHE_TTL_SECONDS` | `3600` | Cache entry TTL in seconds |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model used for semantic embeddings |
| `COMPRESSION_THRESHOLD` | `0.80` | Compress when X% of context window is used |
| `COMPRESSION_MODEL` | `gpt-4o-mini` | Cheap model used for summarisation |

---

## Examples

```bash
# Semantic cache demo — shows hit rate across similar questions
uv run python examples/demo_cache.py

# Context compression demo — simulates a long conversation
uv run python examples/demo_compression.py

# Interactive chatbot with both features active
uv run python examples/demo_chatbot.py
```

---

## Tests

```bash
uv run pytest
```

---

## Architecture

```
Request
  │
  ▼
SemanticCache.get(prompt)
  ├─ HIT  → return cached response immediately
  └─ MISS
       │
       ▼
  ContextCompressor.compress(messages, model)
       │   if tokens > 80% of context limit:
       │   summarise old turns → inject summary message
       ▼
  litellm.acompletion(model, messages)
       │
       ▼
  SemanticCache.set(prompt, response)
       │
       ▼
  Return response
```

## How it works

### Semantic Cache
Each prompt is converted to a vector embedding using `text-embedding-3-small`. Incoming prompts are compared against cached embeddings using cosine similarity. If the score exceeds the threshold (default 0.92), the cached response is returned — no LLM call needed.

### Context Compressor
Token usage is tracked per conversation. When usage exceeds `COMPRESSION_THRESHOLD × context_window`, the oldest messages (excluding the system prompt and the most recent exchange) are summarised into a single compact message using a cheap fast model. The conversation continues without truncation.
