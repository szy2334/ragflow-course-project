import asyncio
import json
import time

from fastapi.testclient import TestClient
from sqlalchemy import update

from app.core.config import Settings
from app.db.base import build_engine
from app.db.models import Paper
from app.main import create_app


def _settings(tmp_path):
    return Settings(
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'api.db').as_posix()}",
        object_storage_path=tmp_path / "uploads",
        access_token_secret="test-access-token-secret",
    )


def _headers(token: str | None = None, **extra: str) -> dict[str, str]:
    headers = {"X-Request-Id": "request-test"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    headers.update(extra)
    return headers


def _register(client: TestClient) -> str:
    response = client.post(
        "/api/v1/auth/register",
        headers=_headers(),
        json={
            "email": "person@example.test",
            "password": "a-secure-password",
            "display_name": "Person",
        },
    )
    assert response.status_code == 200
    return response.json()["data"]["token"]["access_token"]


async def _mark_ready(database_url: str, paper_id: str) -> None:
    engine = build_engine(database_url)
    async with engine.begin() as connection:
        await connection.execute(
            update(Paper)
            .where(Paper.paper_id == paper_id)
            .values(status="ready", index_status="succeeded", quality_status="ready")
        )
    await engine.dispose()


def test_write_request_id_and_login_contract(tmp_path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        missing = client.post(
            "/api/v1/auth/register",
            json={
                "email": "person@example.test",
                "password": "a-secure-password",
                "display_name": "Person",
            },
        )
        assert missing.status_code == 400
        assert missing.json()["code"] == "REQUEST_ID_REQUIRED"
        token = _register(client)
        me = client.get("/api/v1/auth/me", headers=_headers(token))
        assert me.status_code == 200
        assert me.json()["data"]["email"] == "person@example.test"
        assert "password" not in json.dumps(me.json())


def test_question_idempotency_and_terminal_sse_resume(tmp_path):
    settings = _settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        token = _register(client)
        upload = client.post(
            "/api/v1/papers",
            headers=_headers(token, **{"Idempotency-Key": "upload-1"}),
            files={"files": ("paper.pdf", b"%PDF-1.4\nminimal", "application/pdf")},
            data={"auto_index": "false"},
        )
        assert upload.status_code == 202
        paper_id = upload.json()["data"]["items"][0]["paper_id"]
        asyncio.run(_mark_ready(settings.database_url, paper_id))
        session = client.post(
            "/api/v1/sessions",
            headers=_headers(token),
            json={"title": "Paper chat", "paper_ids": [paper_id]},
        )
        assert session.status_code == 201
        session_id = session.json()["data"]["session_id"]
        body = {"question": "这篇论文说明了什么？", "stream": True}
        headers = _headers(token, **{"Idempotency-Key": "question-1"})
        accepted = client.post(
            f"/api/v1/sessions/{session_id}/messages", headers=headers, json=body
        )
        assert accepted.status_code == 202
        replay = client.post(f"/api/v1/sessions/{session_id}/messages", headers=headers, json=body)
        assert replay.status_code == 202
        assert replay.json()["data"]["task_id"] == accepted.json()["data"]["task_id"]

        message_id = accepted.json()["data"]["message_id"]
        for _ in range(20):
            task = client.get(accepted.json()["data"]["status_url"], headers=_headers(token))
            if task.json()["data"]["status"] in {"failed", "cancelled"}:
                break
            time.sleep(0.02)
        assert task.json()["data"]["error"]["code"] == "MODEL_NOT_CONFIGURED"
        events = client.get(f"/api/v1/messages/{message_id}/events", headers=_headers(token))
        assert events.status_code == 200
        assert "event: error" in events.text
        event_id = next(line[4:] for line in events.text.splitlines() if line.startswith("id: "))
        resumed = client.get(
            f"/api/v1/messages/{message_id}/events",
            headers=_headers(token, **{"Last-Event-ID": event_id}),
        )
        assert resumed.status_code == 200
        assert resumed.text == ""
