import logging

from pydantic_settings import BaseSettings, SettingsConfigDict

_LOG = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Load and validate application settings from environment variables or a .env file.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM API keys — passed through to litellm via env vars automatically.
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Redis (optional — falls back to in-memory store if empty).
    redis_url: str = ""

    # Semantic cache.
    cache_similarity_threshold: float = 0.92
    cache_ttl_seconds: int = 3600
    embedding_model: str = "text-embedding-3-small"

    # Context compression.
    compression_threshold: float = 0.80
    compression_model: str = "gpt-4o-mini"

    def validate_settings(self) -> None:
        """
        Assert that all settings are within valid ranges.
        """
        assert 0.0 <= self.cache_similarity_threshold <= 1.0, (
            f"cache_similarity_threshold must be in [0.0, 1.0], "
            f"got '{self.cache_similarity_threshold}'"
        )
        assert 0.0 < self.compression_threshold <= 1.0, (
            f"compression_threshold must be in (0.0, 1.0], "
            f"got '{self.compression_threshold}'"
        )
        assert self.cache_ttl_seconds > 0, (
            f"cache_ttl_seconds must be positive, got '{self.cache_ttl_seconds}'"
        )
        if not self.openai_api_key and not self.anthropic_api_key:
            _LOG.warning(
                "No API keys configured — set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env."
            )


settings = Settings()
settings.validate_settings()
