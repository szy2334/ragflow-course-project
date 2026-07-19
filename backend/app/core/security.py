"""Dependency-free password and signed bearer-token helpers."""

import base64
import hashlib
import hmac
import json
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

from .errors import ApiError


def hash_password(password: str) -> str:
    if len(password) < 8:
        raise ApiError(422, "VALIDATION_ERROR", "密码长度至少为 8 位。")
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
    return "scrypt$" + base64.urlsafe_b64encode(salt + digest).decode("ascii")


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, payload = encoded.split("$", 1)
        raw = base64.urlsafe_b64decode(payload.encode("ascii"))
        if scheme != "scrypt" or len(raw) < 17:
            return False
        expected = hashlib.scrypt(password.encode("utf-8"), salt=raw[:16], n=2**14, r=8, p=1)
        return hmac.compare_digest(raw[16:], expected)
    except (ValueError, TypeError):
        return False


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def issue_access_token(
    *, user_id: str, role: str, secret: str, ttl_seconds: int
) -> tuple[str, datetime]:
    expires_at = datetime.now(UTC) + timedelta(seconds=ttl_seconds)
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode("utf-8"))
    payload = _b64(
        json.dumps(
            {"sub": user_id, "role": role, "exp": int(expires_at.timestamp())},
            separators=(",", ":"),
        ).encode("utf-8")
    )
    signature = _b64(
        hmac.new(
            secret.encode("utf-8"), f"{header}.{payload}".encode("ascii"), hashlib.sha256
        ).digest()
    )
    return f"{header}.{payload}.{signature}", expires_at


def decode_access_token(token: str, secret: str) -> dict[str, Any]:
    try:
        header, payload, signature = token.split(".")
        expected = _b64(
            hmac.new(
                secret.encode("utf-8"), f"{header}.{payload}".encode("ascii"), hashlib.sha256
            ).digest()
        )
        if not hmac.compare_digest(signature, expected):
            raise ValueError("signature")
        claims = json.loads(_unb64(payload))
        if not isinstance(claims, dict) or int(claims["exp"]) <= int(datetime.now(UTC).timestamp()):
            raise ValueError("expired")
        if not isinstance(claims.get("sub"), str) or not isinstance(claims.get("role"), str):
            raise ValueError("claims")
        return claims
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise ApiError(401, "UNAUTHENTICATED", "登录状态已失效。") from None


def new_refresh_token() -> str:
    return secrets.token_urlsafe(48)


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
