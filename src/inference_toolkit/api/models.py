from pydantic import BaseModel, Field

# ── Chat completion ───────────────────────────────────────────────────────────


class Message(BaseModel):
    """
    Represent a single message in a chat conversation.
    """

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """
    Accept an OpenAI-compatible chat completion payload.
    """

    model: str
    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False


class CacheHitMeta(BaseModel):
    """
    Explain why a cache hit was returned.
    """

    matched_prompt: str
    similarity_score: float
    cache_age_seconds: float


# ── Cache management ──────────────────────────────────────────────────────────


class CacheStatsResponse(BaseModel):
    """
    Report cache performance metrics for a session.
    """

    total_requests: int
    cache_hits: int
    hit_rate: float = Field(description="Cache hit ratio in the range [0.0, 1.0].")
    total_entries: int


class CacheClearedResponse(BaseModel):
    """
    Confirm that the cache was successfully cleared.
    """

    cleared: bool = True
    message: str = "Cache cleared successfully."


class CacheEntryDetailResponse(BaseModel):
    """
    Expose a single cache entry for the /v1/cache/inspect endpoint.
    """

    prompt_preview: str
    response_preview: str
    model: str
    hits: int
    cost_usd: float
    estimated_savings_usd: float
    age_seconds: float
    created_at: float


class ClusterMemberResponse(BaseModel):
    """
    Represent one entry inside a semantic cluster.
    """

    prompt_preview: str
    hits: int
    similarity_to_centroid: float


class ClusterResponse(BaseModel):
    """
    Group of semantically similar cached prompts.
    """

    cluster_id: int
    centroid_prompt: str
    member_count: int
    total_hits: int
    avg_similarity: float
    members: list[ClusterMemberResponse]


class SavingsReportResponse(BaseModel):
    """
    Summarise estimated cost savings from all cache hits.
    """

    total_entries: int
    total_cache_hits: int
    total_cost_of_original_calls_usd: float
    estimated_savings_usd: float
    avg_cost_per_call_usd: float


# ── Conversations ─────────────────────────────────────────────────────────────


class CreateConversationRequest(BaseModel):
    """
    Request body for creating a new stateful conversation.
    """

    model: str
    system_prompt: str = ""


class ConversationMessageRequest(BaseModel):
    """
    Request body for sending a message to an existing conversation.
    """

    content: str


class ConversationResponse(BaseModel):
    """
    Represent the current state of a conversation.
    """

    id: str
    model: str
    message_count: int
    cumulative_cost_usd: float
    created_at: float


class ConversationTurnResponse(BaseModel):
    """
    Return the result of one conversation turn, including cache and cost metadata.
    """

    response: str
    cached: bool
    cache_meta: CacheHitMeta | None = None
    cost_usd: float
    cumulative_cost_usd: float
    tokens_used: int
    compressed: bool


class ConversationDeletedResponse(BaseModel):
    """
    Confirm that a conversation was deleted.
    """

    deleted: bool = True
    conversation_id: str
