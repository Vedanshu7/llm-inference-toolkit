import logging

import litellm
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from inference_toolkit.api.models import (
    CacheClearedResponse,
    CacheStatsResponse,
    ChatCompletionRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest) -> JSONResponse:
    cache: object = request.app.state.cache
    compressor: object = request.app.state.compressor

    last_user_message = next(
        (m.content for m in reversed(body.messages) if m.role == "user"),
        None,
    )

    # 1. Check semantic cache
    if last_user_message:
        cached = await cache.get(last_user_message)
        if cached:
            logger.info("Returning cached response for: %s", last_user_message[:60])
            return JSONResponse(
                content={
                    "id": "cached",
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

    # 2. Compress context if needed
    messages_dicts = [m.model_dump() for m in body.messages]
    try:
        messages_dicts = await compressor.compress(messages_dicts, body.model)
    except Exception as exc:
        logger.warning("Context compression failed, continuing without it: %s", exc)

    # 3. Call LLM via litellm
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
        logger.error("LLM call failed: %s", exc)
        raise HTTPException(status_code=502, detail="LLM provider error") from exc

    # 4. Cache the response
    assistant_content = response.choices[0].message.content
    if last_user_message and assistant_content:
        await cache.set(last_user_message, assistant_content)

    return JSONResponse(content=response.model_dump())


@router.get("/v1/cache/stats", response_model=CacheStatsResponse)
async def cache_stats(request: Request) -> CacheStatsResponse:
    stats = await request.app.state.cache.stats()
    return CacheStatsResponse(
        total_requests=stats.total_requests,
        cache_hits=stats.cache_hits,
        hit_rate=stats.hit_rate,
        total_entries=stats.total_entries,
    )


@router.delete("/v1/cache", response_model=CacheClearedResponse)
async def clear_cache(request: Request) -> CacheClearedResponse:
    await request.app.state.cache.clear()
    return CacheClearedResponse()
