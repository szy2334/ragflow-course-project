# Backend Runtime

The FastAPI service is exposed at `/api/v1`. It owns authentication,
ownership-scoped resources, idempotent task acceptance, task status and
resumable SSE. The existing `app.ai` package remains the sole AI workflow
entry point; the runtime injects its storage and external-service ports.

Run locally with a fresh environment file:

```powershell
Copy-Item .env.example .env
python -m pip install -e ".[test]"
uvicorn app.main:app --reload --port 8000
```

For local API-contract testing, SQLite and the in-memory event store are
allowed when `APP_ENV=development`. In production the service refuses to
start unless `DATABASE_URL` is PostgreSQL, `REDIS_URL` is set, and
`ACCESS_TOKEN_SECRET` is present.

## Runtime Boundaries

- PostgreSQL stores users, refresh tokens, papers, structured chunks,
  sessions, messages, task records, citations, reviews,
  configuration snapshots, workflow runs and node traces.
- Redis stores task snapshots, cancellation markers and a bounded event stream
  per message. `Last-Event-ID` and `after_sequence` both resume the SSE feed.
- MinerU and Baidu specialized OCR are invoked only through adapters. User PDFs
  remain local after parsing; RAGFlow is queried only for fixed reference-paper
  evidence during question answering.
- Model configuration is read from environment variables. API keys are not
  accepted from clients or persisted in workflow configuration snapshots.

`POST /sessions/{session_id}/messages` creates a distinct `task_id` and
`message_id`. The client can poll `/tasks/{task_id}` and connect to
`/messages/{message_id}/events`. Only validated, committed results emit
`delta` and `final` events.

## Database Schema and Migrations

The document-aligned schema is defined in `app/db/models.py`; production uses
Alembic as its only schema creation and upgrade path. Apply migrations before
starting the production API:

```powershell
$env:DATABASE_URL = "postgresql+asyncpg://user:password@host/database"
python -m alembic upgrade head
python -m alembic check
```

For a local migration contract check, point `DATABASE_URL` to a disposable
SQLite database. `python -m alembic downgrade base` fully removes the initial
schema. Development and tests may still bootstrap an empty SQLite database
through ORM metadata; production startup never calls `create_all`. The backend
CI also runs the migration round trip against an ephemeral PostgreSQL service.
