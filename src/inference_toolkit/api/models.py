from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False


class CacheStatsResponse(BaseModel):
    total_requests: int
    cache_hits: int
    hit_rate: float = Field(description="0.0–1.0")
    total_entries: int


class CacheClearedResponse(BaseModel):
    cleared: bool = True
    message: str = "Cache cleared successfully"
