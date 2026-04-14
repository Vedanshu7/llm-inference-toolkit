from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM API keys — passed through to litellm via env vars automatically
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Redis (optional — falls back to in-memory store if empty)
    redis_url: str = ""

    # Semantic cache
    cache_similarity_threshold: float = 0.92
    cache_ttl_seconds: int = 3600
    embedding_model: str = "text-embedding-3-small"

    # Context compression
    compression_threshold: float = 0.80
    compression_model: str = "gpt-4o-mini"


settings = Settings()
