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
  RAGFlow mappings, sessions, messages, task records, citations, reviews,
  configuration snapshots, workflow runs and node traces.
- Redis stores task snapshots, cancellation markers and a bounded event stream
  per message. `Last-Event-ID` and `after_sequence` both resume the SSE feed.
- MinerU, Baidu specialized OCR and RAGFlow are invoked only through adapters.
  Required OCR or a partial RAGFlow mapping fails the paper task; it never
  advances the paper to `ready`.
- Model configuration is read from environment variables. API keys are not
  accepted from clients or persisted in workflow configuration snapshots.

`POST /sessions/{session_id}/messages` creates a distinct `task_id` and
`message_id`. The client can poll `/tasks/{task_id}` and connect to
`/messages/{message_id}/events`. Only validated, committed results emit
`delta` and `final` events.
