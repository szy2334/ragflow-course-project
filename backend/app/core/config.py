"""Environment-only runtime configuration.

No credential has a source-code default. Development can use SQLite and the
in-memory event store, while production requires PostgreSQL and Redis.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    api_prefix: str = "/api/v1"
    database_url: str = "sqlite+aiosqlite:///./paper_review.db"
    redis_url: str | None = None
    redis_event_ttl_seconds: int = 86_400
    redis_event_maxlen: int = 2_000
    access_token_secret: SecretStr | None = None
    access_token_ttl_seconds: int = 900
    refresh_token_ttl_seconds: int = 604_800
    cors_origins: list[str] = ["http://localhost:5173"]
    max_upload_mb: int = 50
    object_storage_path: Path = Path("./var/uploads")
    ragflow_base_url: str | None = None
    ragflow_api_key: SecretStr | None = None
    ragflow_reference_dataset_id: str | None = None
    # Backward compatibility for deployments configured before user PDFs stopped being indexed.
    ragflow_user_dataset_id: str | None = None
    ragflow_public_dataset_id: str | None = None
    mineru_base_url: str | None = None
    mineru_api_key: SecretStr | None = None
    baidu_ocr_base_url: str | None = None
    baidu_ocr_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    llm_model: str = ""
    model_config_version: str = "env-v1"
    graph_version: str = "1.0"
    prompt_version: str = "v1"
    schema_version: str = "v1"
    standard_version: str | None = None

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in {"production", "prod"}

    @property
    def ragflow_reference_dataset(self) -> str | None:
        return self.ragflow_reference_dataset_id or self.ragflow_user_dataset_id

    def validate_runtime(self) -> None:
        if self.is_production and not self.database_url.startswith("postgresql+"):
            raise RuntimeError("DATABASE_URL must use PostgreSQL in production")
        if self.is_production and not self.redis_url:
            raise RuntimeError("REDIS_URL is required in production")
        if self.is_production and not self.access_token_secret:
            raise RuntimeError("ACCESS_TOKEN_SECRET is required in production")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_runtime()
    return settings
