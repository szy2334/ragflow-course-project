"""Database-backed exactly-once response replay for externally retried writes."""

# ruff: noqa: E501

import hashlib
import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ApiError
from app.db.models import IdempotencyRecord


def request_fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


async def replay_or_raise(
    session: AsyncSession,
    *,
    user_id: str,
    key: str,
    method: str,
    path: str,
    fingerprint: str,
) -> dict[str, Any] | None:
    record = await session.scalar(
        select(IdempotencyRecord).where(
            IdempotencyRecord.user_id == user_id,
            IdempotencyRecord.key == key,
            IdempotencyRecord.method == method,
            IdempotencyRecord.path == path,
        )
    )
    if record is None:
        return None
    if record.request_hash != fingerprint:
        raise ApiError(409, "IDEMPOTENCY_CONFLICT", "幂等键已用于不同请求。")
    return record.response_json


def save_response(
    session: AsyncSession,
    *,
    user_id: str,
    key: str,
    method: str,
    path: str,
    fingerprint: str,
    status_code: int,
    response: dict[str, Any],
) -> None:
    session.add(
        IdempotencyRecord(
            user_id=user_id,
            key=key,
            method=method,
            path=path,
            request_hash=fingerprint,
            status_code=status_code,
            response_json=response,
        )
    )
