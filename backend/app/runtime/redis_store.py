"""Redis-backed task, cancellation and resumable SSE event store."""

import asyncio
import json
from collections import defaultdict
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.ai.schemas import StreamEvent
from app.core.config import Settings


class RedisRuntime:
    """Uses Redis when configured; in-memory fallback is development-only."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._redis: Redis | None = None
        self._memory_tasks: dict[str, dict[str, Any]] = {}
        self._memory_cancelled: set[str] = set()
        self._memory_events: dict[str, list[StreamEvent]] = defaultdict(list)
        self._memory_event_ids: dict[str, int] = {}
        self._lock = asyncio.Lock()

    @property
    def persistent(self) -> bool:
        return self._redis is not None

    async def connect(self) -> None:
        if not self._settings.redis_url:
            if self._settings.is_production:
                raise RuntimeError("REDIS_URL is required in production")
            return
        client = Redis.from_url(self._settings.redis_url, decode_responses=True)
        try:
            await client.ping()
        except RedisError:
            await client.aclose()
            if self._settings.is_production:
                raise RuntimeError("Redis is unavailable in production") from None
            return
        self._redis = client

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def set_task_state(self, task_id: str, state: dict[str, Any]) -> None:
        if self._redis is not None:
            await self._redis.set(
                f"task:{task_id}",
                json.dumps(state, ensure_ascii=False, default=str),
                ex=self._settings.redis_event_ttl_seconds,
            )
            return
        async with self._lock:
            self._memory_tasks[task_id] = state

    async def get_task_state(self, task_id: str) -> dict[str, Any] | None:
        if self._redis is not None:
            value = await self._redis.get(f"task:{task_id}")
            return json.loads(value) if value else None
        async with self._lock:
            return self._memory_tasks.get(task_id)

    async def cancel(self, task_id: str) -> None:
        if self._redis is not None:
            await self._redis.set(
                f"cancel:{task_id}", "1", ex=self._settings.redis_event_ttl_seconds
            )
            return
        async with self._lock:
            self._memory_cancelled.add(task_id)

    async def is_cancelled(self, task_id: str) -> bool:
        if self._redis is not None:
            return bool(await self._redis.exists(f"cancel:{task_id}"))
        async with self._lock:
            return task_id in self._memory_cancelled

    async def append_event(self, event: StreamEvent) -> None:
        payload = event.model_dump_json()
        if self._redis is not None:
            pipe = self._redis.pipeline(transaction=True)
            pipe.xadd(
                f"sse:{event.message_id}",
                {"sequence": str(event.sequence), "payload": payload},
                maxlen=self._settings.redis_event_maxlen,
                approximate=True,
            )
            pipe.set(
                f"sse-event:{event.message_id}:{event.event_id}",
                str(event.sequence),
                ex=self._settings.redis_event_ttl_seconds,
            )
            pipe.expire(f"sse:{event.message_id}", self._settings.redis_event_ttl_seconds)
            await pipe.execute()
            return
        async with self._lock:
            events = self._memory_events[event.message_id]
            events.append(event)
            self._memory_event_ids[f"{event.message_id}:{event.event_id}"] = event.sequence
            if len(events) > self._settings.redis_event_maxlen:
                dropped = events.pop(0)
                self._memory_event_ids.pop(f"{event.message_id}:{dropped.event_id}", None)

    async def after_event_id(self, message_id: str, event_id: str | None) -> int | None:
        if not event_id:
            return None
        if self._redis is not None:
            value = await self._redis.get(f"sse-event:{message_id}:{event_id}")
            return int(value) if value else None
        async with self._lock:
            return self._memory_event_ids.get(f"{message_id}:{event_id}")

    async def events_after(self, message_id: str, sequence: int) -> list[StreamEvent]:
        if self._redis is not None:
            rows = await self._redis.xrange(f"sse:{message_id}", min="-", max="+")
            events: list[StreamEvent] = []
            for _, fields in rows:
                value = fields.get("payload")
                if not value:
                    continue
                event = StreamEvent.model_validate_json(value)
                if event.sequence > sequence:
                    events.append(event)
            return events
        async with self._lock:
            return [event for event in self._memory_events[message_id] if event.sequence > sequence]

    async def latest_event(self, message_id: str) -> StreamEvent | None:
        events = await self.events_after(message_id, 0)
        return events[-1] if events else None
