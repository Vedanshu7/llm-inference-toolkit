from pydantic import BaseModel, Field


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
