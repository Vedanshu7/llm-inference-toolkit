import logging
import uuid

import litellm
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

import inference_toolkit.api.models as api_models
from inference_toolkit.conversation.manager import BudgetExceededError

_LOG = logging.getLogger(__name__)
router = APIRouter()


# ── Stateless chat completions ────────────────────────────────────────────────


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request, body: api_models.ChatCompletionRequest
) -> JSONResponse:
    """
    Handle a stateless chat completion request with semantic caching and context compression.

    Check the semantic cache first; on a miss, compress context if needed, call the
    upstream LLM, and store the response in the cache.

    :param request: FastAPI request carrying app state (cache, compressor)
    :param body: OpenAI-compatible chat completion payload
    :return: chat completion response — includes cache_meta on a hit
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
        hit = await cache.get(last_user_message)
        if hit:
            _LOG.info(
                "[%s] Cache hit (score=%.4f) for: %s",
                request_id,
                hit.similarity_score,
                last_user_message[:60],
            )
            return JSONResponse(
                content={
                    "id": f"cached-{request_id}",
                    "object": "chat.completion",
                    "model": body.model,
                    "cached": True,
                    "cache_meta": {
                        "matched_prompt": hit.matched_prompt,
                        "similarity_score": hit.similarity_score,
                        "cache_age_seconds": hit.cache_age_seconds,
                    },
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": hit.response},
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
    kwargs: dict[str, object] = {"model": body.model, "messages": messages_dicts}
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
    # Cache the response for future similar prompts, including cost metadata.
    assistant_content = str(response.choices[0].message.content)
    if last_user_message and assistant_content:
        try:
            cost = float(litellm.completion_cost(completion_response=response) or 0.0)
        except Exception:
            cost = 0.0
        await cache.set(last_user_message, assistant_content, model=body.model, cost_usd=cost)
    return JSONResponse(content=response.model_dump())


# ── Cache management ──────────────────────────────────────────────────────────


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


@router.get("/v1/cache/inspect", response_model=list[api_models.CacheEntryDetailResponse])
async def cache_inspect(request: Request) -> list[api_models.CacheEntryDetailResponse]:
    """
    Return a detailed view of every entry in the cache, sorted by hit count.

    :param request: FastAPI request carrying app state
    :return: list of entry details with cost and savings estimates
    """
    details = await request.app.state.analytics.inspect(request.app.state.cache._store)
    return [
        api_models.CacheEntryDetailResponse(
            prompt_preview=d.prompt_preview,
            response_preview=d.response_preview,
            model=d.model,
            hits=d.hits,
            cost_usd=d.cost_usd,
            estimated_savings_usd=d.estimated_savings_usd,
            age_seconds=d.age_seconds,
            created_at=d.created_at,
        )
        for d in details
    ]


@router.get("/v1/cache/clusters", response_model=list[api_models.ClusterResponse])
async def cache_clusters(
    request: Request, threshold: float = 0.85
) -> list[api_models.ClusterResponse]:
    """
    Group cached prompts by semantic similarity and return the cluster report.

    :param request: FastAPI request carrying app state
    :param threshold: cosine similarity threshold for cluster membership
    :return: list of clusters sorted by total hits descending
    """
    clusters = await request.app.state.analytics.clusters(
        request.app.state.cache._store, threshold=threshold
    )
    return [
        api_models.ClusterResponse(
            cluster_id=c.cluster_id,
            centroid_prompt=c.centroid_prompt,
            member_count=c.member_count,
            total_hits=c.total_hits,
            avg_similarity=c.avg_similarity,
            members=[
                api_models.ClusterMemberResponse(
                    prompt_preview=m.prompt_preview,
                    hits=m.hits,
                    similarity_to_centroid=m.similarity_to_centroid,
                )
                for m in c.members
            ],
        )
        for c in clusters
    ]


@router.get("/v1/cache/savings", response_model=api_models.SavingsReportResponse)
async def cache_savings(request: Request) -> api_models.SavingsReportResponse:
    """
    Return an estimated cost savings report for all cached entries.

    :param request: FastAPI request carrying app state
    :return: total savings, hit count, and average cost per call
    """
    report = await request.app.state.analytics.savings_report(request.app.state.cache._store)
    return api_models.SavingsReportResponse(
        total_entries=report.total_entries,
        total_cache_hits=report.total_cache_hits,
        total_cost_of_original_calls_usd=report.total_cost_of_original_calls_usd,
        estimated_savings_usd=report.estimated_savings_usd,
        avg_cost_per_call_usd=report.avg_cost_per_call_usd,
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


# ── Stateful conversations ────────────────────────────────────────────────────


@router.post("/v1/conversations", response_model=api_models.ConversationResponse)
async def create_conversation(
    request: Request, body: api_models.CreateConversationRequest
) -> api_models.ConversationResponse:
    """
    Create a new stateful conversation.

    :param request: FastAPI request carrying app state
    :param body: model and optional system prompt
    :return: conversation metadata including the assigned ID
    """
    conv_store = request.app.state.conv_store
    conversation = await conv_store.create(model=body.model, system_prompt=body.system_prompt)
    return api_models.ConversationResponse(
        id=conversation.id,
        model=conversation.model,
        message_count=len(conversation.messages),
        cumulative_cost_usd=conversation.cumulative_cost_usd,
        created_at=conversation.created_at,
    )


@router.post(
    "/v1/conversations/{conversation_id}/messages",
    response_model=api_models.ConversationTurnResponse,
)
async def send_message(
    request: Request,
    conversation_id: str,
    body: api_models.ConversationMessageRequest,
) -> api_models.ConversationTurnResponse:
    """
    Send a message to an existing conversation and return the assistant's response.

    Applies semantic caching, cost guardrails, and context compression automatically.

    :param request: FastAPI request carrying app state
    :param conversation_id: UUID of the target conversation
    :param body: user message content
    :return: turn result with response, cache metadata, cost, and compression flag
    """
    conv_store = request.app.state.conv_store
    conversation = await conv_store.get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")
    try:
        turn = await conversation.send(
            prompt=body.content,
            cache=request.app.state.cache,
            compressor=request.app.state.compressor,
        )
    except BudgetExceededError as exc:
        raise HTTPException(
            status_code=402,
            detail=f"Budget exceeded: ${exc.cumulative_cost:.4f} >= ${exc.max_cost:.4f}",
        ) from exc
    await conv_store.save(conversation)
    # Build cache_meta only when a hit occurred.
    cache_meta = None
    if turn.cache_hit:
        cache_meta = api_models.CacheHitMeta(
            matched_prompt=turn.cache_hit.matched_prompt,
            similarity_score=turn.cache_hit.similarity_score,
            cache_age_seconds=turn.cache_hit.cache_age_seconds,
        )
    return api_models.ConversationTurnResponse(
        response=turn.response,
        cached=turn.cache_hit is not None,
        cache_meta=cache_meta,
        cost_usd=turn.cost_usd,
        cumulative_cost_usd=turn.cumulative_cost_usd,
        tokens_used=turn.tokens_used,
        compressed=turn.compressed,
    )


@router.get("/v1/conversations/{conversation_id}", response_model=api_models.ConversationResponse)
async def get_conversation(
    request: Request, conversation_id: str
) -> api_models.ConversationResponse:
    """
    Retrieve the current state of a conversation.

    :param request: FastAPI request carrying app state
    :param conversation_id: UUID of the conversation
    :return: conversation metadata
    """
    conversation = await request.app.state.conv_store.get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")
    return api_models.ConversationResponse(
        id=conversation.id,
        model=conversation.model,
        message_count=len(conversation.messages),
        cumulative_cost_usd=conversation.cumulative_cost_usd,
        created_at=conversation.created_at,
    )


@router.delete(
    "/v1/conversations/{conversation_id}",
    response_model=api_models.ConversationDeletedResponse,
)
async def delete_conversation(
    request: Request, conversation_id: str
) -> api_models.ConversationDeletedResponse:
    """
    Delete a conversation and all its history.

    :param request: FastAPI request carrying app state
    :param conversation_id: UUID of the conversation to delete
    :return: deletion confirmation
    """
    deleted = await request.app.state.conv_store.delete(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")
    return api_models.ConversationDeletedResponse(conversation_id=conversation_id)
