"""LangGraph node implementations for the complete QA/review workflow."""

import asyncio
import time
from collections.abc import Awaitable
from typing import Any

from ..agents import ControllerAgent, PaperUnderstandingAgent, ReviewAgentA, ReviewAgentB
from ..errors import ModelOutputInvalid, WorkflowCancelled
from ..ports import WorkflowDependencies
from ..schemas import (
    AnswerDraft,
    AnswerView,
    ConfigurationSnapshot,
    EvidenceItem,
    NodeTrace,
    PaperUnderstanding,
    RetrieveEvidenceRequest,
    RetrieveStandardsRequest,
    ReviewGraphState,
    ReviewOpinions,
    RouteDecision,
    StartQaWorkflowCommand,
    ValidationResult,
)
from ..validators import AnswerValidationPipeline
from .events import WorkflowEventEmitter
from .policies import WorkflowPolicy


class WorkflowNodes:
    def __init__(
        self,
        *,
        dependencies: WorkflowDependencies,
        controller: ControllerAgent,
        paper_agent: PaperUnderstandingAgent,
        review_a: ReviewAgentA,
        review_b: ReviewAgentB,
        validators: AnswerValidationPipeline,
        events: WorkflowEventEmitter,
        policy: WorkflowPolicy,
    ) -> None:
        self._deps = dependencies
        self._controller = controller
        self._paper_agent = paper_agent
        self._review_a = review_a
        self._review_b = review_b
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

    async def route(self, state: ReviewGraphState) -> dict[str, Any]:
        command = _command(state)
        started = await self._begin(state, "routing", "正在识别问题意图")
        decision, result = await self._controller.route(
            original_question=command.original_question,
            conversation_summary=state.get("conversation_summary", ""),
            configuration=command.configuration,
        )
        await self._check_cancelled(command)
        warnings = [*state.get("warnings", []), *decision.warnings]
        await self._trace(
            command,
            "route",
            started,
            "succeeded",
            metrics=result.metrics.model_dump(mode="json"),
        )
        return {
            "route_decision": decision.model_dump(mode="json"),
            "agent_results": [*state.get("agent_results", []), result.model_dump(mode="json")],
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
            await self._trace(command, "paper_retrieve", started, "failed", "RAG_NO_EVIDENCE")
            return {
                "paper_evidences": [],
                "warnings": [*warnings, "paper retrieval failed"],
                "error_code": "RAG_NO_EVIDENCE",
                "error_message": "论文中未找到足够信息。",
                "sequence": self._events.sequence,
            }

        allowed: list[EvidenceItem] = []
        for item in evidence_set.items:
            if item.source_type != "paper" or item.paper_id not in command.paper_ids:
                warnings.append(f"discarded out-of-scope retrieval item {item.evidence_id}")
                continue
            allowed.append(item)
        if not allowed:
            await self._trace(command, "paper_retrieve", started, "failed", "RAG_NO_EVIDENCE")
            return {
                "paper_evidences": [],
                "warnings": warnings,
                "error_code": "RAG_NO_EVIDENCE",
                "error_message": "论文中未找到足够信息。",
                "sequence": self._events.sequence,
            }
        for item in allowed:
            await self._events.emit("citation", {"evidence": item.model_dump(mode="json")})
        await self._trace(command, "paper_retrieve", started, "succeeded")
        return {
            "paper_evidences": [item.model_dump(mode="json") for item in allowed],
            "warnings": warnings,
            "error_code": None,
            "error_message": None,
            "sequence": self._events.sequence,
        }

    async def paper_understand(self, state: ReviewGraphState) -> dict[str, Any]:
        command = _command(state)
        route = _route(state)
        evidences = _evidences(state, "paper_evidences")
        started = await self._begin(state, "understanding", "正在提取论文事实")
        try:
            understanding, result = await self._with_model_timeout(
                command.configuration,
                self._paper_agent.run(
                    standalone_question=route.standalone_question,
                    evidences=evidences,
                    configuration=command.configuration,
                ),
            )
            allowed_ids = {item.evidence_id for item in evidences}
            cited_ids = {item for fact in understanding.facts for item in fact.evidence_ids}
            if not cited_ids.issubset(allowed_ids):
                raise ModelOutputInvalid("paper agent cited evidence outside the current context")
            await self._check_cancelled(command)
        except Exception as exc:
            if isinstance(exc, WorkflowCancelled):
                raise
            await self._trace(
                command, "paper_understand", started, "failed", "MODEL_OUTPUT_INVALID"
            )
            return {
                "paper_understanding": None,
                "error_code": "MODEL_OUTPUT_INVALID",
                "error_message": str(exc),
                "warnings": [*state.get("warnings", []), "paper understanding failed"],
                "sequence": self._events.sequence,
            }
        await self._trace(
            command,
            "paper_understand",
            started,
            "succeeded",
            metrics=result.metrics.model_dump(mode="json"),
        )
        return {
            "paper_understanding": understanding.model_dump(mode="json"),
            "agent_results": [*state.get("agent_results", []), result.model_dump(mode="json")],
            "error_code": None,
            "error_message": None,
            "sequence": self._events.sequence,
        }

    async def standard_retrieve(self, state: ReviewGraphState) -> dict[str, Any]:
        command = _command(state)
        route = _route(state)
        started = await self._begin(state, "retrieving_standards", "正在检索评审标准")
        warnings = list(state.get("warnings", []))
        request = RetrieveStandardsRequest(
            task_id=command.task_id,
            standalone_question=route.standalone_question,
            route_type=route.effective_route_type,
            dimensions=route.review_dimensions,
            standard_version=command.configuration.standard_version,
        )
        try:
            async with asyncio.timeout(self._policy.retrieval_timeout_seconds):
                evidence_set = await self._deps.retrieval.retrieve_standards(request)
            standards = [item for item in evidence_set.items if item.source_type == "standard"]
            warnings.extend(evidence_set.warnings)
            await self._check_cancelled(command)
        except Exception as exc:
            if isinstance(exc, WorkflowCancelled):
                raise
            standards = []
        if not standards:
            warnings.append("PUBLIC_KB_UNAVAILABLE: review is limited to paper evidence")
            await self._trace(
                command, "standard_retrieve", started, "failed", "PUBLIC_KB_UNAVAILABLE"
            )
            return {
                "standard_evidences": [],
                "skip_reviews": True,
                "warnings": warnings,
                "error_code": (
                    "PUBLIC_KB_UNAVAILABLE" if route.effective_route_type == "score" else None
                ),
                "error_message": (
                    "评分所需的公共标准不可用。"
                    if route.effective_route_type == "score"
                    else None
                ),
                "sequence": self._events.sequence,
            }
        for item in standards:
            await self._events.emit("citation", {"evidence": item.model_dump(mode="json")})
        await self._trace(command, "standard_retrieve", started, "succeeded")
        return {
            "standard_evidences": [item.model_dump(mode="json") for item in standards],
            "skip_reviews": False,
            "warnings": warnings,
            "sequence": self._events.sequence,
        }

    async def review_a(self, state: ReviewGraphState) -> dict[str, Any]:
        command = _command(state)
        route = _route(state)
        understanding = _understanding(state)
        started = await self._begin(state, "review_a", "评审 A 正在严格审查")
        try:
            reviews, result = await self._with_model_timeout(
                command.configuration,
                self._review_a.run(
                    standalone_question=route.standalone_question,
                    dimensions=route.review_dimensions,
                    paper_understanding=understanding,
                    paper_evidences=_evidences(state, "paper_evidences"),
                    standard_evidences=_evidences(state, "standard_evidences"),
                    configuration=command.configuration,
                ),
            )
            _validate_review_evidence(reviews, state)
            await self._check_cancelled(command)
        except Exception as exc:
            if isinstance(exc, WorkflowCancelled):
                raise
            await self._trace(command, "review_a", started, "failed", "MODEL_OUTPUT_INVALID")
            return {
                "review_a": None,
                "warnings": [*state.get("warnings", []), f"review A unavailable: {exc}"],
                "sequence": self._events.sequence,
            }
        await self._trace(
            command,
            "review_a",
            started,
            "succeeded",
            metrics=result.metrics.model_dump(mode="json"),
        )
        return {
            "review_a": reviews.model_dump(mode="json"),
            "agent_results": [*state.get("agent_results", []), result.model_dump(mode="json")],
            "sequence": self._events.sequence,
        }

    async def review_b(self, state: ReviewGraphState) -> dict[str, Any]:
        command = _command(state)
        route = _route(state)
        understanding = _understanding(state)
        review_a = (
            ReviewOpinions.model_validate(state["review_a"])
            if state.get("review_a")
            else None
        )
        started = await self._begin(state, "review_b", "评审 B 正在交叉核验")
        try:
            reviews, result = await self._with_model_timeout(
                command.configuration,
                self._review_b.run(
                    standalone_question=route.standalone_question,
                    dimensions=route.review_dimensions,
                    paper_understanding=understanding,
                    review_a=review_a,
                    paper_evidences=_evidences(state, "paper_evidences"),
                    standard_evidences=_evidences(state, "standard_evidences"),
                    configuration=command.configuration,
                ),
            )
            _validate_review_evidence(reviews, state)
            await self._check_cancelled(command)
        except Exception as exc:
            if isinstance(exc, WorkflowCancelled):
                raise
            await self._trace(command, "review_b", started, "failed", "MODEL_OUTPUT_INVALID")
            return {
                "review_b": None,
                "warnings": [*state.get("warnings", []), f"review B unavailable: {exc}"],
                "sequence": self._events.sequence,
            }
        await self._trace(
            command,
            "review_b",
            started,
            "succeeded",
            metrics=result.metrics.model_dump(mode="json"),
        )
        return {
            "review_b": reviews.model_dump(mode="json"),
            "agent_results": [*state.get("agent_results", []), result.model_dump(mode="json")],
            "sequence": self._events.sequence,
        }

    async def synthesize(self, state: ReviewGraphState) -> dict[str, Any]:
        command = _command(state)
        route = _route(state)
        started = await self._begin(state, "synthesizing", "正在汇总已核验证据")
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
            draft, result = await self._with_model_timeout(
                command.configuration,
                self._controller.synthesize(
                    original_question=command.original_question,
                    route=route,
                    paper_understanding=_understanding(state),
                    review_a=(
                        ReviewOpinions.model_validate(state["review_a"])
                        if state.get("review_a")
                        else None
                    ),
                    review_b=(
                        ReviewOpinions.model_validate(state["review_b"])
                        if state.get("review_b")
                        else None
                    ),
                    evidences=[
                        *_evidences(state, "paper_evidences"),
                        *_evidences(state, "standard_evidences"),
                    ],
                    warnings=state.get("warnings", []),
                    configuration=command.configuration,
                    previous_draft=previous,
                    validation_errors=validation.errors if validation else [],
                ),
            )
            await self._check_cancelled(command)
        except Exception as exc:
            if isinstance(exc, WorkflowCancelled):
                raise
            await self._trace(command, "synthesize", started, "failed", "MODEL_OUTPUT_INVALID")
            return {
                "error_code": "MODEL_OUTPUT_INVALID",
                "error_message": str(exc),
                "warnings": [*state.get("warnings", []), "answer synthesis failed"],
                "sequence": self._events.sequence,
            }
        await self._trace(
            command,
            "synthesize",
            started,
            "succeeded",
            metrics=result.metrics.model_dump(mode="json"),
        )
        return {
            "draft_answer": draft.model_dump(mode="json"),
            "agent_results": [*state.get("agent_results", []), result.model_dump(mode="json")],
            "error_code": None,
            "error_message": None,
            "sequence": self._events.sequence,
        }

    async def validate(self, state: ReviewGraphState) -> dict[str, Any]:
        command = _command(state)
        started = await self._begin(state, "validating", "正在核验引用与数字")
        draft = AnswerDraft.model_validate(state["draft_answer"])
        result = self._validators.validate(
            draft=draft,
            route=_route(state),
            understanding=(
                PaperUnderstanding.model_validate(state["paper_understanding"])
                if state.get("paper_understanding")
                else None
            ),
            evidences=[
                *_evidences(state, "paper_evidences"),
                *_evidences(state, "standard_evidences"),
            ],
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

    async def safe_refusal(self, state: ReviewGraphState) -> dict[str, Any]:
        route = _route(state)
        await self._begin(state, "refusing", "现有证据不足，正在生成保守说明")
        code = state.get("error_code")
        if route.effective_route_type == "out_of_scope":
            reason = "问题超出当前论文问答与评审范围。"
        elif code == "RAG_NO_EVIDENCE":
            reason = "当前论文中未找到足以回答该问题的证据。"
        elif code == "MODEL_OUTPUT_INVALID":
            reason = "系统未能生成通过结构与证据校验的回答。"
        elif code == "PUBLIC_KB_UNAVAILABLE":
            reason = "评分所需的公共评审标准不可用，无法给出可靠分数。"
        else:
            reason = "现有证据不足，无法给出可靠结论。"
        warnings = list(state.get("warnings", []))
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
            is_refusal=True,
            refusal_reason=reason,
        )
        return {
            "draft_answer": draft.model_dump(mode="json"),
            "validation": ValidationResult(valid=True).model_dump(mode="json"),
            "sequence": self._events.sequence,
        }

    async def finalize(self, state: ReviewGraphState) -> dict[str, Any]:
        command = _command(state)
        started = await self._begin(state, "finalizing", "回答已通过校验")
        draft = AnswerDraft.model_validate(state["draft_answer"])
        evidence_by_id = {
            item.evidence_id: item
            for item in [
                *_evidences(state, "paper_evidences"),
                *_evidences(state, "standard_evidences"),
            ]
        }
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
            score=draft.score,
            confidence=draft.confidence,
            warnings=draft.warnings,
            is_refusal=draft.is_refusal,
            refusal_reason=draft.refusal_reason,
        )
        if state.get("review_a") or state.get("review_b"):
            await self._events.emit(
                "review_summary",
                {
                    "review_a": state.get("review_a"),
                    "review_b": state.get("review_b"),
                    "warnings": state.get("warnings", []),
                },
            )
        for offset in range(0, len(answer.answer), self._policy.delta_chunk_size):
            await self._events.emit(
                "delta",
                {"delta": answer.answer[offset : offset + self._policy.delta_chunk_size]},
            )
        await self._events.emit("final", {"answer": answer.model_dump(mode="json")})
        await self._trace(command, "finalize", started, "succeeded")
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


def _understanding(state: ReviewGraphState) -> PaperUnderstanding:
    value = state.get("paper_understanding")
    if value is None:
        raise ModelOutputInvalid("paper understanding is unavailable")
    return PaperUnderstanding.model_validate(value)


def _evidences(state: ReviewGraphState, key: str) -> list[EvidenceItem]:
    return [EvidenceItem.model_validate(item) for item in state.get(key, [])]


def _validate_review_evidence(reviews: ReviewOpinions, state: ReviewGraphState) -> None:
    paper_ids = {item.evidence_id for item in _evidences(state, "paper_evidences")}
    standard_ids = {item.evidence_id for item in _evidences(state, "standard_evidences")}
    for opinion in reviews.opinions:
        for claim in opinion.claims:
            if not set(claim.paper_evidence_ids).issubset(paper_ids):
                raise ModelOutputInvalid("review cited paper evidence outside current context")
            if not set(claim.standard_evidence_ids).issubset(standard_ids):
                raise ModelOutputInvalid("review cited standard evidence outside current context")


def _content_preferences(question: str) -> list[str]:
    normalized = question.lower()
    preferences: list[str] = []
    if any(token in normalized for token in ("表", "table", "数值")):
        preferences.append("table")
    if any(token in normalized for token in ("图", "figure", "流程图")):
        preferences.extend(["figure", "figure_caption"])
    if any(token in normalized for token in ("公式", "formula", "loss")):
        preferences.append("formula")
    return list(dict.fromkeys(preferences))
