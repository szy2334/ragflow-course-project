"""Environment-only runtime configuration.

No credential has a source-code default. Development can use SQLite and the
in-memory event store, while production requires PostgreSQL and Redis.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
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
    # Read-only source for preprocessed reference-paper artifacts.  Keep it
    # separate from user uploads, which remain ownership-scoped above.
    user_paper_root: Path | None = None
    ragflow_base_url: str | None = None
    ragflow_api_key: SecretStr | None = None
    mineru_base_url: str | None = None
    # MinerU names this credential MINERU_TOKEN; accept the legacy API-key
    # spelling as well so deployments do not need to duplicate a secret.
    mineru_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("MINERU_TOKEN", "MINERU_API_KEY")
    )
    baidu_ocr_base_url: str | None = None
    baidu_ocr_api_key: SecretStr | None = None
    baidu_ocr_secret_key: SecretStr | None = None
    baidu_ocr_token_url: str = "https://aip.baidubce.com/oauth/2.0/token"
    baidu_ocr_accurate_url: str = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"
    baidu_ocr_table_url: str = "https://aip.baidubce.com/rest/2.0/ocr/v1/table"
    baidu_ocr_paddle_task_url: str = (
        "https://aip.baidubce.com/rest/2.0/brain/online/v2/paddle-vl-parser/task"
    )
    baidu_ocr_paddle_query_url: str = (
        "https://aip.baidubce.com/rest/2.0/brain/online/v2/paddle-vl-parser/task/query"
    )
    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    llm_model: str = ""
    llm_timeout_seconds: float = Field(default=120.0, gt=0, le=300)
    llm_structured_mode: Literal["json_schema", "json_object", "prompt_json"] = "json_schema"
    model_config_version: str = "env-v1"
    graph_version: str = "1.0"
    prompt_version: str = "v1"
    schema_version: str = "v1"

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
    def user_paper_runs_path(self) -> Path | None:
        if self.user_paper_root is None:
            return None
        return self.user_paper_root.resolve() / "runs"

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
