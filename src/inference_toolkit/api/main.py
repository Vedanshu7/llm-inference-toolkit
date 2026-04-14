import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import litellm
from fastapi import FastAPI

import inference_toolkit.api.routes as api_routes
import inference_toolkit.cache.analytics as cache_analytics_module
import inference_toolkit.cache.semantic_cache as semantic_cache_module
import inference_toolkit.cache.store as cache_store
import inference_toolkit.compression.compressor as compressor_module
import inference_toolkit.conversation.manager as conversation_module
from inference_toolkit.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
_LOG = logging.getLogger(__name__)

# Register API keys with litellm via environment variables.
if settings.openai_api_key:
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
if settings.anthropic_api_key:
    os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)

# Silently drop unsupported parameters per provider rather than raising.
litellm.drop_params = True


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage startup and shutdown of shared app state (cache, compressor).
    """
    store = await cache_store.get_store(settings.redis_url)
    app.state.cache = semantic_cache_module.SemanticCache(store)
    app.state.compressor = compressor_module.ContextCompressor()
    app.state.analytics = cache_analytics_module.CacheAnalytics()
    app.state.conv_store = conversation_module.ConversationStore()
    backend = "Redis" if settings.redis_url else "in-memory"
    _LOG.info("Inference toolkit started (cache backend=%s).", backend)
    yield
    _LOG.info("Inference toolkit shutting down.")


app = FastAPI(
    title="LLM Inference Toolkit",
    description="Semantic response cache + context compression engine for LLM inference",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(api_routes.router)


@app.get("/health")
async def health() -> dict[str, str]:
    """
    Return a liveness check payload.

    :return: dict with status key
    """
    return {"status": "ok"}


def main() -> None:
    """
    Launch the uvicorn server as an installed script entry point.
    """
    import uvicorn

    uvicorn.run("inference_toolkit.api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
