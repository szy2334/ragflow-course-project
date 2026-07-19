"""Production FastAPI application factory."""

import secrets
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.core.errors import ApiError, api_error_handler
from app.db.base import build_engine, build_session_factory
from app.db.models import Base
from app.runtime.executor import WorkflowTaskExecutor
from app.runtime.redis_store import RedisRuntime
from app.workers.ingestion import IngestionTaskExecutor
from app.workers.operations import OperationsTaskExecutor


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = build_engine(runtime_settings.database_url)
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        app.state.settings = runtime_settings
        app.state.engine = engine
        app.state.session_factory = build_session_factory(engine)
        app.state.access_token_secret = (
            runtime_settings.access_token_secret.get_secret_value()
            if runtime_settings.access_token_secret
            else secrets.token_urlsafe(48)
        )
        app.state.redis = RedisRuntime(runtime_settings)
        await app.state.redis.connect()
        app.state.workflow_executor = WorkflowTaskExecutor(
            runtime_settings, app.state.session_factory, app.state.redis
        )
        app.state.ingestion_executor = IngestionTaskExecutor(
            runtime_settings, app.state.session_factory, app.state.redis
        )
        app.state.operations_executor = OperationsTaskExecutor(
            runtime_settings, app.state.session_factory, app.state.redis
        )
        yield
        await app.state.redis.close()
        await engine.dispose()

    app = FastAPI(
        title="科研论文智能阅读系统",
        version="1.0",
        openapi_url=f"{runtime_settings.api_prefix}/openapi.json",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=runtime_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Request-Id",
            "Idempotency-Key",
            "If-Match",
            "Last-Event-ID",
        ],
    )

    @app.middleware("http")
    async def correlation_id(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-Id") or str(uuid4())
        response = await call_next(request)
        response.headers["X-Request-Id"] = request.state.request_id
        return response

    app.add_exception_handler(ApiError, api_error_handler)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "code": "VALIDATION_ERROR",
                "message": "请求参数不符合规范。",
                "details": {"errors": exc.errors()},
                "request_id": getattr(request.state, "request_id", str(uuid4())),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "code": "INTERNAL_ERROR",
                "message": "服务暂时不可用，请稍后重试。",
                "request_id": getattr(request.state, "request_id", str(uuid4())),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

    app.include_router(router, prefix=runtime_settings.api_prefix)
    return app


app = create_app()
