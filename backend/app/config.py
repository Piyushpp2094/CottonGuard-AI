"""
Environment-backed application configuration using pydantic-settings.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Typed runtime settings loaded from environment variables and .env."""

    APP_NAME: str = "CottonGuard AI"
    APP_VERSION: str = "2.1.0"
    ENVIRONMENT: str = "local"

    SECRET_KEY: str = "cotton-leaf-ai-dev-fallback-key-change-in-env"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    COOKIE_SECURE: bool = False
    COOKIE_SAMESITE: str = "strict"

    DATABASE_URL: str = f"sqlite:///{PROJECT_ROOT / 'cotton_leaf_ai.db'}"

    MAX_UPLOAD_SIZE_MB: int = 10
    MIN_IMAGE_DIM: int = 100
    MAX_IMAGE_DIM: int = 8000

    UPLOAD_DIR: str = str(PROJECT_ROOT / "artifacts" / "predictions")
    ARTIFACTS_DIR: str = str(PROJECT_ROOT / "artifacts")

    SSE_TIMEOUT_SECONDS: int = 180

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def max_upload_bytes(self) -> int:
        return int(self.MAX_UPLOAD_SIZE_MB) * 1024 * 1024

    def resolve_artifacts_dir(self) -> Path:
        p = Path(self.ARTIFACTS_DIR)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    @property
    def resolved_database_url(self) -> str:
        url = self.DATABASE_URL
        if url.startswith("sqlite:///./"):
            rel = url[len("sqlite:///./"):]
            return f"sqlite:///{(PROJECT_ROOT / rel).resolve()}"
        return url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns cached Settings instance; call get_settings.cache_clear() in tests if env changes."""
    return Settings()
