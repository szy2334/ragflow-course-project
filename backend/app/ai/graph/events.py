"""Sequenced event emission independent of the HTTP/SSE transport."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ..ports import EventSink
from ..schemas import StartQaWorkflowCommand, StreamEvent


class WorkflowEventEmitter:
    def __init__(self, sink: EventSink, command: StartQaWorkflowCommand) -> None:
        self._sink = sink
        self._command = command
        self._sequence = 0
        self._terminal = False

    @property
    def sequence(self) -> int:
        return self._sequence

    @property
    def terminal(self) -> bool:
        return self._terminal

    def synchronize(self, sequence: int) -> None:
        self._sequence = max(self._sequence, sequence)

    async def emit(self, event_type: str, data: dict[str, Any]) -> None:
        if self._terminal:
            raise RuntimeError("cannot emit a business event after a terminal event")
        self._sequence += 1
        event = StreamEvent(
            event_id=str(uuid4()),
            event_type=event_type,
            task_id=self._command.task_id,
            message_id=self._command.message_id,
            session_id=self._command.session_id,
            sequence=self._sequence,
            timestamp=datetime.now(UTC),
            data=data,
        )
        await self._sink.emit(event)
        if event_type in {"final", "error"}:
            self._terminal = True
