import logging
import os
from contextlib import asynccontextmanager

import litellm
from fastapi import FastAPI

from inference_toolkit.api.routes import router
from inference_toolkit.cache.semantic_cache import SemanticCache
from inference_toolkit.cache.store import get_store
from inference_toolkit.compression.compressor import ContextCompressor
from inference_toolkit.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

# Pass API keys to litellm via environment (litellm reads these automatically)
if settings.openai_api_key:
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
if settings.anthropic_api_key:
    os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)

litellm.drop_params = True  # silently ignore unsupported params per provider


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = await get_store(settings.redis_url)
    app.state.cache = SemanticCache(store)
    app.state.compressor = ContextCompressor()
    backend = "Redis" if settings.redis_url else "in-memory"
    logger.info("Inference toolkit started (cache backend=%s)", backend)
    yield
    logger.info("Inference toolkit shutting down")


app = FastAPI(
    title="LLM Inference Toolkit",
    description="Semantic response cache + context compression engine for LLM inference",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def main() -> None:
    import uvicorn
    uvicorn.run("inference_toolkit.api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
