# ── Stage 1: build deps with uv ─────────────────────────────────────────────
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Install deps into an isolated venv (no project source needed yet)
COPY pyproject.toml .
RUN uv sync --frozen --no-dev --no-install-project

# ── Stage 2: lean runtime image ─────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ /app/src/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "inference_toolkit.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
