"""Stable external API errors and request-correlation middleware."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import Request
from fastapi.responses import JSONResponse


class ApiError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


def request_id_for(request: Request) -> str:
    return getattr(request.state, "request_id", request.headers.get("X-Request-Id") or str(uuid4()))


def envelope(data: Any, request_id: str, *, message: str = "") -> dict[str, Any]:
    return {
        "code": "ok",
        "message": message,
        "data": data,
        "request_id": request_id,
        "timestamp": datetime.now(UTC).isoformat(),
    }


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.code,
            "message": exc.message,
            "details": exc.details,
            "request_id": request_id_for(request),
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )
