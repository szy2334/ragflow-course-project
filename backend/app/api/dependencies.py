"""FastAPI dependencies for authenticated ownership-scoped access."""

# ruff: noqa: B008

from collections.abc import AsyncIterator

from fastapi import Depends, Header, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ApiError, request_id_for
from app.core.security import decode_access_token
from app.db.models import User

bearer = HTTPBearer(auto_error=False)


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    factory = request.app.state.session_factory
    async with factory() as session:
        yield session


async def current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    session: AsyncSession = Depends(get_session),
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise ApiError(401, "UNAUTHENTICATED", "请先登录。")
    claims = decode_access_token(credentials.credentials, request.app.state.access_token_secret)
    user = await session.get(User, claims["sub"])
    if user is None or not user.is_active:
        raise ApiError(401, "UNAUTHENTICATED", "登录状态已失效。")
    return user


def require_request_id(
    request: Request,
    request_id: str | None = Header(default=None, alias="X-Request-Id"),
) -> str:
    if not request_id or not request_id.strip() or len(request_id) > 128:
        raise ApiError(400, "REQUEST_ID_REQUIRED", "写请求必须携带 X-Request-Id。")
    return request_id_for(request)


def require_idempotency_key(
    key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> str:
    if not key or not key.strip() or len(key) > 255:
        raise ApiError(400, "IDEMPOTENCY_KEY_REQUIRED", "该请求必须携带 Idempotency-Key。")
    return key


def require_admin(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise ApiError(403, "FORBIDDEN", "需要管理员权限。")
    return user
