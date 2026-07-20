import asyncio
import json
import time

from fastapi.testclient import TestClient
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import Settings
from app.db.base import build_engine
from app.db.models import ConfigurationRevision, Paper, PaperVersion, SessionPaper, User
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


async def _promote_admin(database_url: str, user_id: str) -> None:
    engine = build_engine(database_url)
    async with engine.begin() as connection:
        await connection.execute(update(User).where(User.user_id == user_id).values(role="admin"))
    await engine.dispose()


async def _row_count(database_url: str, model) -> int:
    engine = build_engine(database_url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with sessions() as session:
        count = await session.scalar(select(func.count()).select_from(model))
    await engine.dispose()
    return count or 0


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
        paper_detail = client.get(f"/api/v1/papers/{paper_id}", headers=_headers(token))
        assert paper_detail.status_code == 200
        assert "understanding" in paper_detail.json()["data"]
        asyncio.run(_mark_ready(settings.database_url, paper_id))
        session = client.post(
            "/api/v1/sessions",
            headers=_headers(token, **{"Idempotency-Key": "session-1"}),
            json={"title": "Paper chat", "paper_ids": [paper_id]},
        )
        assert session.status_code == 201
        session_replay = client.post(
            "/api/v1/sessions",
            headers=_headers(token, **{"Idempotency-Key": "session-1"}),
            json={"title": "Paper chat", "paper_ids": [paper_id]},
        )
        assert session_replay.status_code == 201
        assert session_replay.json()["data"]["session_id"] == session.json()["data"]["session_id"]
        assert asyncio.run(_row_count(settings.database_url, PaperVersion)) == 1
        assert asyncio.run(_row_count(settings.database_url, SessionPaper)) == 1
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
        feedback = client.post(
            f"/api/v1/messages/{message_id}/feedback",
            headers=_headers(token, **{"Idempotency-Key": "feedback-1"}),
            json={"feedback_type": "issue", "reason": "citation mismatch"},
        )
        assert feedback.status_code == 200
        assert feedback.json()["data"]["feedback_type"] == "issue"
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


def test_admin_configuration_and_evaluation_contract(tmp_path):
    settings = _settings(tmp_path)
    app = create_app(settings)
    configuration_id = "00000000-0000-4000-8000-000000000001"
    with TestClient(app) as client:
        token = _register(client)
        user_id = client.get("/api/v1/auth/me", headers=_headers(token)).json()["data"]["user_id"]
        asyncio.run(_promote_admin(settings.database_url, user_id))

        denied = client.get("/api/v1/admin/model-configs", headers=_headers("invalid"))
        assert denied.status_code == 401

        create = client.put(
            f"/api/v1/admin/model-configs/{configuration_id}",
            headers=_headers(
                token,
                **{"Idempotency-Key": "config-create", "If-Match": "*"},
            ),
            json={"value": {"name": "primary_generation", "model_name": "test-model"}},
        )
        assert create.status_code == 201
        assert create.json()["data"]["model_config_id"] == configuration_id
        version = create.headers["etag"]

        listed = client.get("/api/v1/admin/model-configs", headers=_headers(token))
        assert listed.status_code == 200
        assert listed.json()["data"]["total"] == 1

        conflict = client.put(
            f"/api/v1/admin/model-configs/{configuration_id}",
            headers=_headers(
                token,
                **{"Idempotency-Key": "config-conflict", "If-Match": "outdated"},
            ),
            json={"value": {"name": "primary_generation", "model_name": "new-model"}},
        )
        assert conflict.status_code == 409
        assert conflict.json()["code"] == "CONFIG_VERSION_CONFLICT"

        update_config = client.put(
            f"/api/v1/admin/model-configs/{configuration_id}",
            headers=_headers(
                token,
                **{"Idempotency-Key": "config-update", "If-Match": version},
            ),
            json={"value": {"name": "primary_generation", "model_name": "new-model"}},
        )
        assert update_config.status_code == 200
        assert update_config.headers["etag"] != version
        assert asyncio.run(_row_count(settings.database_url, ConfigurationRevision)) == 2

        evaluation = client.post(
            "/api/v1/admin/evaluation-runs",
            headers=_headers(token, **{"Idempotency-Key": "evaluation-1"}),
            json={"dataset_id": "qasper", "experiment_type": "multi_agent_rag"},
        )
        assert evaluation.status_code == 202
        task_url = evaluation.json()["data"]["status_url"]
        for _ in range(20):
            task = client.get(task_url, headers=_headers(token))
            if task.json()["data"]["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.02)
        assert task.json()["data"]["status"] == "succeeded"
        assert task.json()["data"]["result"]["dataset_id"] == "qasper"

        metrics = client.get("/api/v1/admin/metrics/overview", headers=_headers(token))
        assert metrics.status_code == 200
        assert metrics.json()["data"]["request_count"] >= 1
