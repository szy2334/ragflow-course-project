"""Deterministic orchestration nodes for the V3 paper-reading workflow."""

import asyncio
import time
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import Any

from ..agents import AnswerGeneratorAgent, IntentRouterAgent, fallback_route
from ..errors import AiWorkflowError, AnswerPersistenceFailed, WorkflowCancelled
from ..ports import WorkflowDependencies
from ..schemas import (
    AgentResult,
    AnswerDraft,
    AnswerView,
    ConfigurationSnapshot,
    EvidenceItem,
    NodeTrace,
    PersistAnswerCommand,
    RetrieveEvidenceRequest,
    ReviewGraphState,
    RouteDecision,
    StartQaWorkflowCommand,
    ValidationResult,
)
from ..validators import AnswerValidationPipeline
from .events import WorkflowEventEmitter
from .policies import WorkflowPolicy


class MasterController:
    """Orchestrate tasks without making any model call itself."""

    def __init__(
        self,
        *,
        dependencies: WorkflowDependencies,
        intent_router: IntentRouterAgent,
        answer_generator: AnswerGeneratorAgent,
        validators: AnswerValidationPipeline,
        events: WorkflowEventEmitter,
        policy: WorkflowPolicy,
    ) -> None:
        self._deps = dependencies
        self._intent_router = intent_router
        self._answer_generator = answer_generator
        self._validators = validators
        self._events = events
        self._policy = policy

    async def load_context(self, state: ReviewGraphState) -> dict[str, Any]:
        command = _command(state)
        started = await self._begin(state, "loading_context", "正在加载会话上下文")
        warnings = list(state.get("warnings", []))
        try:
            async with asyncio.timeout(self._policy.context_timeout_seconds):
                summary = await self._deps.context.load_session_summary(
                    user_id=command.user_id,
                    session_id=command.session_id,
                )
            await self._check_cancelled(command)
            await self._trace(command, "load_context", started, "succeeded")
        except Exception as exc:
            if isinstance(exc, WorkflowCancelled):
                raise
            summary = ""
            warnings.append("session context unavailable; follow-up resolution may be limited")
            await self._trace(command, "load_context", started, "failed", "CONTEXT_UNAVAILABLE")
        return {
            "conversation_summary": summary,
            "warnings": warnings,
            "sequence": self._events.sequence,
        }

    async def intent_route(self, state: ReviewGraphState) -> dict[str, Any]:
        command = _command(state)
        started = await self._begin(state, "routing", "正在识别问题意图")
        try:
            decision, result = await self._with_model_timeout(
                command.configuration,
                self._intent_router.run(
                    original_question=command.original_question,
                    conversation_summary=state.get("conversation_summary", ""),
                    configuration=command.configuration,
                ),
            )
        except AiWorkflowError:
            decision = fallback_route(
                command.original_question,
                state.get("conversation_summary", ""),
            )
            result = None
        await self._check_cancelled(command)
        warnings = [*state.get("warnings", []), *decision.warnings]
        await self._trace(
            command,
            "intent_route",
            started,
            "succeeded",
            metrics=result.metrics.model_dump(mode="json") if result else {},
        )
        agent_results = list(state.get("agent_results", []))
        if result is not None:
            agent_results.append(result.model_dump(mode="json"))
        return {
            "route_decision": decision.model_dump(mode="json"),
            "agent_results": agent_results,
            "warnings": warnings,
            "sequence": self._events.sequence,
        }

    async def paper_retrieve(self, state: ReviewGraphState) -> dict[str, Any]:
        command = _command(state)
        route = _route(state)
        started = await self._begin(state, "retrieving_paper", "正在检索论文证据")
        warnings = list(state.get("warnings", []))
        try:
            evidence_set = await self._retrieve_paper(command, route, relaxed=False)
            if not evidence_set.items:
                warnings.append("paper retrieval returned no evidence; relaxed retry used")
                evidence_set = await self._retrieve_paper(command, route, relaxed=True)
            warnings.extend(evidence_set.warnings)
            await self._check_cancelled(command)
        except Exception as exc:
            if isinstance(exc, WorkflowCancelled):
                raise
            evidence_set = None

        allowed = [] if evidence_set is None else [
            item
            for item in evidence_set.items
            if item.source_type == "paper" and item.paper_id in command.paper_ids
        ]
        if not allowed:
            await self._trace(command, "paper_retrieve", started, "failed", "RAG_NO_EVIDENCE")
            return {
                "paper_evidences": [],
                "paper_summary": evidence_set.paper_summary if evidence_set else "",
                "warnings": warnings,
                "error_code": "RAG_NO_EVIDENCE",
                "error_message": "当前论文没有可用于回答的文本证据。",
                "sequence": self._events.sequence,
            }
        for item in allowed:
            await self._events.emit("citation", {"evidence": item.model_dump(mode="json")})
        await self._trace(command, "paper_retrieve", started, "succeeded")
        return {
            "paper_evidences": [item.model_dump(mode="json") for item in allowed],
            "paper_summary": evidence_set.paper_summary,
            "warnings": warnings,
            "error_code": None,
            "error_message": None,
            "sequence": self._events.sequence,
        }

    async def generate_answer(self, state: ReviewGraphState) -> dict[str, Any]:
        command = _command(state)
        route = _route(state)
        started = await self._begin(state, "generating_answer", "正在基于论文原文生成回答")
        previous = (
            AnswerDraft.model_validate(state["draft_answer"])
            if state.get("draft_answer")
            else None
        )
        validation = (
            ValidationResult.model_validate(state["validation"])
            if state.get("validation")
            else None
        )
        try:
            draft, result, _ = await self._with_model_timeout(
                command.configuration,
                self._answer_generator.run(
                    original_question=command.original_question,
                    route=route,
                    evidences=_evidences(state),
                    paper_summary=state.get("paper_summary", ""),
                    warnings=state.get("warnings", []),
                    configuration=command.configuration,
                    previous_draft=previous,
                    validation_errors=validation.errors if validation else [],
                    # Keep the draft private until deterministic validation and
                    # persistence both succeed. Otherwise a repaired response
                    # would leave an invalid first draft visible in the SSE stream.
                    on_answer_delta=None,
                ),
            )
            await self._check_cancelled(command)
            draft = draft.model_copy(
                update={
                    "warnings": list(
                        dict.fromkeys([*state.get("warnings", []), *draft.warnings])
                    )
                }
            )
        except Exception as exc:
            if isinstance(exc, WorkflowCancelled):
                raise
            await self._trace(command, "generate_answer", started, "failed", "MODEL_OUTPUT_INVALID")
            return {
                "error_code": "MODEL_OUTPUT_INVALID",
                "error_message": str(exc),
                "warnings": [*state.get("warnings", []), "answer generation failed"],
                "sequence": self._events.sequence,
            }
        await self._trace(
            command,
            "generate_answer",
            started,
            "succeeded",
            metrics=result.metrics.model_dump(mode="json"),
        )
        return {
            "draft_answer": draft.model_dump(mode="json"),
            "agent_results": [*state.get("agent_results", []), result.model_dump(mode="json")],
            "error_code": None,
            "error_message": None,
            "answer_streamed": False,
            "sequence": self._events.sequence,
        }

    async def validate(self, state: ReviewGraphState) -> dict[str, Any]:
        command = _command(state)
        started = await self._begin(state, "validating", "正在校验引用与数字")
        result = self._validators.validate(
            draft=AnswerDraft.model_validate(state["draft_answer"]),
            route=_route(state),
            understanding=None,
            evidences=_evidences(state),
            original_question=command.original_question,
        )
        repair_count = state.get("repair_count", 0) + (0 if result.valid else 1)
        terminal_invalid = not result.valid and repair_count > self._policy.max_semantic_repairs
        await self._trace(
            command,
            "validate",
            started,
            "succeeded" if result.valid else "retried",
            None if result.valid else "QA_EVIDENCE_INVALID",
            metrics={"errors": result.errors},
        )
        return {
            "validation": result.model_dump(mode="json"),
            "repair_count": repair_count,
            "error_code": "MODEL_OUTPUT_INVALID" if terminal_invalid else None,
            "error_message": "answer failed semantic validation" if terminal_invalid else None,
            "warnings": (
                [*state.get("warnings", []), *result.errors]
                if terminal_invalid
                else state.get("warnings", [])
            ),
            "sequence": self._events.sequence,
        }

    async def refuse_out_of_scope(self, state: ReviewGraphState) -> dict[str, Any]:
        return await self._refusal(
            state,
            reason="该问题超出当前论文阅读问答范围。",
            node="refuse_out_of_scope",
        )

    async def refuse_failed(self, state: ReviewGraphState) -> dict[str, Any]:
        code = state.get("error_code")
        reason = {
            "RAG_NO_EVIDENCE": "当前论文没有可用于回答该问题的文本证据。",
            "MODEL_OUTPUT_INVALID": "系统未能生成通过引用与数字校验的回答。",
        }.get(code, "当前证据不足，无法生成可靠回答。")
        return await self._refusal(state, reason=reason, node="refuse_failed")

    async def _refusal(
        self,
        state: ReviewGraphState,
        *,
        reason: str,
        node: str,
    ) -> dict[str, Any]:
        command = _command(state)
        route = _route(state)
        started = await self._begin(state, "refusing", "正在生成保守说明")
        warnings = list(state.get("warnings", []))
        code = state.get("error_code")
        if code and code not in warnings:
            warnings.append(code)
        draft = AnswerDraft(
            route_type=route.initial_route_type,
            answer=reason,
            claims=[],
            evidence_ids=[],
            score=None,
            confidence=1.0,
            warnings=warnings,
            evidence_sufficient=False,
            evidence_gap_reason=reason,
            is_refusal=True,
            refusal_reason=reason,
        )
        await self._trace(command, node, started, "succeeded", error_code=code)
        return {
            "draft_answer": draft.model_dump(mode="json"),
            "validation": ValidationResult(valid=True).model_dump(mode="json"),
            "sequence": self._events.sequence,
        }

    async def finalize(self, state: ReviewGraphState) -> dict[str, Any]:
        return await self._finalize(state, insufficient=False)

    async def finalize_insufficient(self, state: ReviewGraphState) -> dict[str, Any]:
        return await self._finalize(state, insufficient=True)

    async def _finalize(
        self,
        state: ReviewGraphState,
        *,
        insufficient: bool,
    ) -> dict[str, Any]:
        command = _command(state)
        node = "finalize_insufficient" if insufficient else "finalize"
        started = await self._begin(state, "finalizing", "回答已通过校验")
        draft = AnswerDraft.model_validate(state["draft_answer"])
        warnings = list(draft.warnings)
        if insufficient and "EVIDENCE_INSUFFICIENT" not in warnings:
            warnings.append("EVIDENCE_INSUFFICIENT")
        evidence_by_id = {item.evidence_id: item for item in _evidences(state)}
        answer = AnswerView(
            message_id=command.message_id,
            session_id=command.session_id,
            task_id=command.task_id,
            route_type=draft.route_type,
            answer=draft.answer,
            claims=draft.claims,
            evidences=[
                evidence_by_id[item] for item in draft.evidence_ids if item in evidence_by_id
            ],
            score=None,
            review_opinions=[],
            standards=[],
            confidence=draft.confidence,
            warnings=warnings,
            evidence_sufficient=draft.evidence_sufficient,
            evidence_gap_reason=draft.evidence_gap_reason,
            is_refusal=draft.is_refusal,
            refusal_reason=draft.refusal_reason,
            completed_at=datetime.now(UTC),
        )
        agent_results = [
            AgentResult.model_validate(item) for item in state.get("agent_results", [])
        ]
        try:
            await self._deps.persistence.persist(
                PersistAnswerCommand(
                    request_id=command.request_id,
                    correlation_id=command.correlation_id,
                    user_id=command.user_id,
                    session_id=command.session_id,
                    task_id=command.task_id,
                    message_id=command.message_id,
                    answer=answer,
                    agent_results=agent_results,
                    configuration=command.configuration,
                )
            )
        except Exception as exc:
            await self._trace(command, node, started, "failed", "ANSWER_PERSIST_FAILED")
            raise AnswerPersistenceFailed("final answer persistence failed") from exc
        if not state.get("answer_streamed"):
            for offset in range(0, len(answer.answer), self._policy.delta_chunk_size):
                await self._events.emit(
                    "delta",
                    {"delta": answer.answer[offset : offset + self._policy.delta_chunk_size]},
                )
        await self._events.emit("final", {"answer": answer.model_dump(mode="json")})
        await self._trace(command, node, started, "succeeded")
        return {
            "final_answer": answer.model_dump(mode="json"),
            "sequence": self._events.sequence,
        }

    async def _retrieve_paper(
        self,
        command: StartQaWorkflowCommand,
        route: RouteDecision,
        *,
        relaxed: bool,
    ):
        request = RetrieveEvidenceRequest(
            task_id=command.task_id,
            user_id=command.user_id,
            paper_ids=command.paper_ids,
            standalone_question=route.standalone_question,
            route_type=route.effective_route_type,
            content_preferences=_content_preferences(route.standalone_question),
            relaxed=relaxed,
        )
        async with asyncio.timeout(self._policy.retrieval_timeout_seconds):
            return await self._deps.retrieval.retrieve_paper(request)

    async def _begin(self, state: ReviewGraphState, stage: str, label: str) -> float:
        self._events.synchronize(state.get("sequence", 0))
        command = _command(state)
        await self._check_cancelled(command)
        await self._events.emit("status", {"stage": stage, "label": label})
        return time.perf_counter()

    async def _check_cancelled(self, command: StartQaWorkflowCommand) -> None:
        if await self._deps.cancellation.is_cancelled(command.task_id):
            raise WorkflowCancelled("workflow was cancelled")

    async def _with_model_timeout(
        self,
        configuration: ConfigurationSnapshot,
        awaitable: Awaitable[Any],
    ) -> Any:
        async with asyncio.timeout(configuration.model.timeout_seconds):
            return await awaitable

    async def _trace(
        self,
        command: StartQaWorkflowCommand,
        node: str,
        started: float,
        status: str,
        error_code: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        trace = NodeTrace(
            request_id=command.request_id,
            correlation_id=command.correlation_id,
            task_id=command.task_id,
            message_id=command.message_id,
            node_name=node,
            duration_ms=int((time.perf_counter() - started) * 1000),
            status=status,
            error_code=error_code,
            metrics=metrics or {},
        )
        try:
            await self._deps.trace.record(trace)
        except Exception:
            return


def _command(state: ReviewGraphState) -> StartQaWorkflowCommand:
    return StartQaWorkflowCommand.model_validate(state["command"])


def _route(state: ReviewGraphState) -> RouteDecision:
    return RouteDecision.model_validate(state["route_decision"])


def _evidences(state: ReviewGraphState) -> list[EvidenceItem]:
    return [
        EvidenceItem.model_validate(item) for item in state.get("paper_evidences", [])
    ]


def _content_preferences(question: str) -> list[str]:
    normalized = question.lower()
    preferences: list[str] = []
    if any(token in normalized for token in ("表", "table", "数值")):
        preferences.append("table")
    if any(token in normalized for token in ("图", "figure", "流程图")):
        preferences.extend(["figure", "figure_caption"])
    if any(token in normalized for token in ("公式", "方程", "formula", "equation")):
        preferences.append("formula")
    return list(dict.fromkeys(preferences))
