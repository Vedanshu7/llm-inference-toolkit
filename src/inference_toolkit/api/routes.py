import logging
import uuid

import litellm
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

import inference_toolkit.api.models as api_models

_LOG = logging.getLogger(__name__)
router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request, body: api_models.ChatCompletionRequest
) -> JSONResponse:
    """
    Handle a chat completion request with semantic caching and context compression.

    Checks the semantic cache first; on a miss, compresses context if needed,
    then calls the upstream LLM and stores the response in the cache.

    :param request: FastAPI request carrying app state (cache, compressor)
    :param body: OpenAI-compatible chat completion payload
    :return: chat completion response, with a `cached: true` flag on hits
    """
    request_id = str(uuid.uuid4())[:8]
    cache = request.app.state.cache
    compressor = request.app.state.compressor
    last_user_message = next(
        (m.content for m in reversed(body.messages) if m.role == "user"),
        None,
    )
    # Check the semantic cache before making any LLM call.
    if last_user_message:
        cached = await cache.get(last_user_message)
        if cached:
            _LOG.info("[%s] Cache hit for: %s", request_id, last_user_message[:60])
            return JSONResponse(
                content={
                    "id": f"cached-{request_id}",
                    "object": "chat.completion",
                    "model": body.model,
                    "cached": True,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": cached.response},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
    # Compress context if the conversation is approaching the model's token limit.
    messages_dicts = [m.model_dump() for m in body.messages]
    try:
        messages_dicts = await compressor.compress(messages_dicts, body.model)
    except Exception as exc:
        _LOG.warning("[%s] Context compression failed, continuing without: %s", request_id, exc)
    # Build kwargs and call the LLM via litellm.
    kwargs: dict = {"model": body.model, "messages": messages_dicts}
    if body.temperature is not None:
        kwargs["temperature"] = body.temperature
    if body.max_tokens is not None:
        kwargs["max_tokens"] = body.max_tokens
    try:
        response = await litellm.acompletion(**kwargs)
    except litellm.exceptions.AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except litellm.exceptions.BadRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _LOG.error("[%s] LLM call failed: %s", request_id, exc)
        raise HTTPException(status_code=502, detail="LLM provider error") from exc
    # Cache the response for future similar prompts.
    assistant_content = response.choices[0].message.content
    if last_user_message and assistant_content:
        await cache.set(last_user_message, assistant_content)
    return JSONResponse(content=response.model_dump())


@router.get("/v1/cache/stats", response_model=api_models.CacheStatsResponse)
async def cache_stats(request: Request) -> api_models.CacheStatsResponse:
    """
    Return current cache performance metrics.

    :param request: FastAPI request carrying app state
    :return: hit rate, total requests, and entry count
    """
    stats = await request.app.state.cache.stats()
    return api_models.CacheStatsResponse(
        total_requests=stats.total_requests,
        cache_hits=stats.cache_hits,
        hit_rate=stats.hit_rate,
        total_entries=stats.total_entries,
    )


@router.delete("/v1/cache", response_model=api_models.CacheClearedResponse)
async def clear_cache(request: Request) -> api_models.CacheClearedResponse:
    """
    Evict all entries from the semantic cache.

    :param request: FastAPI request carrying app state
    :return: confirmation that the cache was cleared
    """
    await request.app.state.cache.clear()
    return api_models.CacheClearedResponse()
