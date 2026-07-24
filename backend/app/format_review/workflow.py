"""LangGraph workflow for composite, evidence-gated format review."""

# ruff: noqa: E501

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any
from uuid import uuid4

import httpx
from langgraph.graph import END, START, StateGraph
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.ai.schemas import StreamEvent
from app.core.config import Settings
from app.db.models import (
    FormatReview,
    FormatReviewItem,
    FormatReviewUnit,
    MediaObjectRecord,
    Paper,
    ParsedBlockRecord,
    PdfTextSpanRecord,
    TaskRecord,
    TraceRecord,
)
from app.runtime.adapters import task_view
from app.runtime.redis_store import RedisRuntime

from .evidence_supplementer import supplement_unit_evidence
from .pdf_layout import bbox_iou, extract_native_pdf_layout
from .runner import FORMAT_REVIEW_INPUT_TOKEN_LIMIT, AgentRunner, format_review_input_tokens
from .schemas import (
    CompositeFormatReviewOutput,
    FormatReviewState,
    FormatSynthesisOutput,
    ReflectionOutput,
)
from .validators import forced_unverifiable_findings, validate_findings
from .venue_layout import (
    SUPPORTED_VENUE_EXTRACTORS,
    VenueLayoutError,
    build_venue_layout_facts,
    locate_mineru_artifact,
    review_facts_from_fused,
)

MAX_RETRIEVAL_ATTEMPTS = 3
MAX_UNIT_CYCLES = 2
# The configured model gateway rate-limits concurrent structured calls. Two
# independent units still progress in parallel without turning a short 429/5xx
# burst into broad "unverifiable" review output.
MAX_UNIT_CONCURRENCY = 2
MAX_BODY_PAGES_PER_UNIT = 2
MAX_CONTEXT_REFINEMENT_DEPTH = 8
# Kept for the unused V1.0 node methods below while historical workflow
# checkpoints remain readable. V1.1 uses the shared unit-cycle budget instead.
MAX_EVIDENCE_RECOVERY = 1
MAX_VALIDATION_REPAIRS = 1
MAX_CONTEXT_CHARACTERS = 100_000
MAX_CONTEXT_FACTS = 56
MAX_COMPOSITE_CONTEXT_FACTS = 48
MAX_CONTEXT_QUOTE_CHARACTERS = 240
FORMAT_CONTEXT_FACTS_PER_CATEGORY = 8
FORMAT_MODEL_INPUT_TOKENS = FORMAT_REVIEW_INPUT_TOKEN_LIMIT


class FormatReviewFailure(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        self.code = code
        self.message = message
        self.retryable = retryable
        super().__init__(code)


class FormatReviewWorkflowService:
    """Runs the dedicated format graph without coupling it to reading QA state."""

    def __init__(
        self,
        settings: Settings,
        sessions: async_sessionmaker[AsyncSession],
        redis: RedisRuntime,
    ) -> None:
        self._settings = settings
        self._sessions = sessions
        self._redis = redis

    async def run(self, task_id: str) -> None:
        async with self._sessions() as session:
            task = await session.get(TaskRecord, task_id)
            thread_resource_id = task.resource_id if task and task.resource_id else task_id
        graph = StateGraph(FormatReviewState)
        graph.add_node("init_review", self.init_review)
        graph.add_node("extract_pdf_layout", self.extract_pdf_layout)
        graph.add_node("build_retrieval_plan", self.build_retrieval_plan)
        graph.add_node("retrieve_standard", self.retrieve_standard)
        graph.add_node("plan_review_units", self.plan_review_units)
        graph.add_node("allocate_rules_to_units", self.allocate_rules_to_units)
        graph.add_node("dispatch_units", self.dispatch_units)
        graph.add_node("synthesize_units", self.synthesize_units)
        graph.add_node("generate_report", self.generate_report)
        graph.add_edge(START, "init_review")
        graph.add_edge("init_review", "extract_pdf_layout")
        graph.add_edge("init_review", "build_retrieval_plan")
        graph.add_edge("extract_pdf_layout", "plan_review_units")
        graph.add_edge("build_retrieval_plan", "retrieve_standard")
        graph.add_edge(["retrieve_standard", "plan_review_units"], "allocate_rules_to_units")
        graph.add_edge("allocate_rules_to_units", "dispatch_units")
        graph.add_edge("dispatch_units", "synthesize_units")
        graph.add_edge("synthesize_units", "generate_report")
        graph.add_edge("generate_report", END)
        await graph.compile().ainvoke(
            {"task_id": task_id, "counters": {}, "metrics": {}, "sequence": 0, "run_events": []},
            config={"configurable": {"thread_id": f"format-review:{thread_resource_id}"}},
        )

    async def init_review(self, state: FormatReviewState) -> dict[str, Any]:
        started = time.perf_counter()
        await self._check_cancelled(state["task_id"])
        async with self._sessions() as session:
            task = await session.get(TaskRecord, state["task_id"])
            review = await session.get(FormatReview, task.resource_id) if task and task.resource_id else None
            paper = await session.get(Paper, review.paper_id) if review else None
            if task is None or review is None:
                raise FormatReviewFailure("FORMAT_REVIEW_NOT_FOUND", "格式审查任务不存在。")
            if paper is None or paper.owner_id != review.user_id or paper.status != "ready":
                raise FormatReviewFailure("PAPER_NOT_READY", "论文尚未完成解析和理解。")
            snapshot = review.profile_snapshot_json if isinstance(review.profile_snapshot_json, dict) else {}
            manifest = snapshot.get("rules") if isinstance(snapshot.get("rules"), list) else []
            if not manifest:
                raise FormatReviewFailure("FORMAT_RULES_UNAVAILABLE", "所选格式规范没有可执行规则。")
            required = (
                snapshot.get("profile_key"),
                snapshot.get("version"),
                snapshot.get("ragflow_dataset_id"),
                snapshot.get("submission_mode"),
            )
            if not all(required):
                raise FormatReviewFailure("FORMAT_PROFILE_SNAPSHOT_INVALID", "格式规范快照不完整，无法继续审查。")
            review.status = "running"
            task.stage, task.progress = "initializing_review", 0.05
            await session.commit()
            task_payload = {
                "task_id": task.task_id,
                "request_id": task.request_id,
                "correlation_id": task.correlation_id,
                "paper_id": paper.paper_id,
                "paper_version_id": paper.paper_version_id,
                "paper_file_path": paper.file_path,
            }
            review_payload = {"format_review_id": review.format_review_id, "submission_mode": review.submission_mode}
        await self._redis.set_task_state(state["task_id"], task_view(task))
        sequence = await self._emit(state, "initializing_review", "正在冻结格式审查依据")
        await self._trace(task_payload, "init_review", started, "succeeded")
        return {"task": task_payload, "review": review_payload, "review_id": review_payload["format_review_id"], "snapshot": snapshot, "sequence": sequence}

    async def extract_pdf_layout(self, state: FormatReviewState) -> dict[str, Any]:
        started = time.perf_counter()
        await self._check_cancelled(state["task_id"])
        await self._set_stage(state, "extracting_layout", 0.20)
        venue_id = str(
            state["snapshot"].get("venue_id") or state["snapshot"].get("profile_key") or ""
        ).lower()
        if venue_id in SUPPORTED_VENUE_EXTRACTORS:
            mineru_artifact = locate_mineru_artifact(
                self._settings.object_storage_path,
                state["task"]["paper_id"],
                state["task"]["paper_version_id"],
            )
            if mineru_artifact is None:
                raise FormatReviewFailure(
                    "FORMAT_VENUE_LAYOUT_INPUT_UNAVAILABLE",
                    "投稿格式解析所需的 MinerU 原始产物不存在，请重新解析论文后再试。",
                    retryable=False,
                )
            try:
                fused_payload = await asyncio.to_thread(
                    build_venue_layout_facts,
                    venue_id=venue_id,
                    pdf_path=Path(state["task"]["paper_file_path"]),
                    mineru_json_path=mineru_artifact,
                )
            except (OSError, ValueError, VenueLayoutError) as exc:
                raise FormatReviewFailure(
                    "FORMAT_VENUE_LAYOUT_FAILED",
                    "所选投稿格式的专用 PDF 解析器执行失败。",
                    retryable=False,
                ) from exc
            facts = review_facts_from_fused(fused_payload)
            if not facts:
                raise FormatReviewFailure(
                    "FORMAT_REVIEW_NO_PAPER_EVIDENCE", "专用解析器未生成可审查的版面事实。"
                )
            quality = {
                **dict(fused_payload.get("quality") or {}),
                "layout_source": "venue_fused_layout",
                "venue_id": venue_id,
                "extractor": SUPPORTED_VENUE_EXTRACTORS[venue_id],
                "mineru_artifact": str(mineru_artifact),
                "schema_version": fused_payload.get("schema_version"),
                "review_fact_count": len(facts),
            }
            await self._emit(state, "extracting_layout", "正在按所选投稿格式提取论文版面事实")
            await self._trace(
                state["task"], "extract_pdf_layout", started, "succeeded", metrics=quality
            )
            return {"layout_facts": facts, "layout_quality": quality}
        async with self._sessions() as session:
            rows = list(
                (
                    await session.scalars(
                        select(ParsedBlockRecord)
                        .where(
                            ParsedBlockRecord.paper_id == state["task"]["paper_id"],
                            ParsedBlockRecord.paper_version_id == state["task"]["paper_version_id"],
                        )
                        .order_by(ParsedBlockRecord.page_number, ParsedBlockRecord.block_id)
                    )
                ).all()
            )
            media_rows = list(
                (
                    await session.scalars(
                        select(MediaObjectRecord)
                        .where(
                            MediaObjectRecord.paper_id == state["task"]["paper_id"],
                            MediaObjectRecord.paper_version_id == state["task"]["paper_version_id"],
                        )
                        .order_by(MediaObjectRecord.page_number, MediaObjectRecord.object_id)
                    )
                ).all()
            )
        native_spans, native_objects, native_quality = await self._load_native_spans(
            paper_id=state["task"]["paper_id"],
            paper_version_id=state["task"]["paper_version_id"],
            file_path=state["task"]["paper_file_path"],
        )
        facts: list[dict[str, Any]] = []
        located = 0
        for index, block in enumerate(rows, start=1):
            # MinerU coordinates have no stable system marker in historical
            # artifacts. They may assist semantic matching but never become a
            # report annotation until a native PDF span confirms the location.
            parsed_bbox = block.bbox_json if _is_bbox(block.bbox_json) else None
            metadata = block.metadata_json if isinstance(block.metadata_json, dict) else {}
            matching_spans = _matching_spans(block.page_number, block.content, parsed_bbox, native_spans)
            # A parsed semantic block may legitimately cover multiple native
            # PDF spans. `_matching_spans` ranks deterministic text/geometry
            # matches, so retain the strongest observable span instead of
            # discarding useful coordinates whenever there is more than one.
            primary_span = matching_spans[0] if matching_spans else None
            matched_style = _matched_block_style(matching_spans)
            bbox = matched_style["bbox"] if matched_style is not None else None
            if bbox:
                located += 1
            facts.append(
                {
                    "evidence_id": f"P{index}",
                    "block_id": block.block_id,
                    "page_number": block.page_number,
                    "bbox": bbox,
                    "quote": block.content,
                    "role": metadata.get("content_role") or block.content_type,
                    "section_title": block.section_title,
                    "font_name": matched_style["font_name"] if matched_style else None,
                    "font_size_pt": matched_style["font_size_pt"] if matched_style else None,
                    "font_size_raw_pt": matched_style["font_size_raw_pt"] if matched_style else None,
                    "font_size_tolerance_pt": 0.25 if matched_style else None,
                    "font_flags": matched_style["font_flags"] if matched_style else None,
                    "alignment": _horizontal_alignment(bbox, primary_span.page_width_pt)
                    if primary_span else metadata.get("alignment"),
                    "is_bold": matched_style["is_bold"] if matched_style else None,
                    "source": "native_pdf+mineru" if primary_span else "mineru_layout",
                    "confidence": metadata.get("confidence"),
                    "page_rotation": primary_span.page_rotation if primary_span else metadata.get("rotation", 0),
                    "source_uri": block.source_ref,
                }
            )
        for span in native_spans:
            facts.append(
                {
                    "evidence_id": f"P{len(facts) + 1}",
                    "block_id": f"native-span-{span.span_index}",
                    "page_number": span.page_number,
                    "bbox": span.bbox_json,
                    "quote": span.text,
                    "role": "native_text_span",
                    "section_title": None,
                    "font_name": span.font_name,
                    "font_size_pt": span.font_size_pt,
                    "font_flags": span.font_flags,
                    "alignment": _horizontal_alignment(span.bbox_json, span.page_width_pt),
                    "is_bold": _font_is_bold(span),
                    "source": span.extraction_source,
                    "confidence": 1.0,
                    "page_rotation": span.page_rotation,
                    "page_width_pt": span.page_width_pt,
                    "page_height_pt": span.page_height_pt,
                    "source_uri": f"paper://{state['task']['paper_id']}/native-span/{span.span_index}",
                }
            )
        for item in native_objects:
            vertical_rules = int(item.get("vertical_rule_count") or 0)
            horizontal_rules = int(item.get("horizontal_rule_count") or 0)
            object_type = str(item["object_type"])
            facts.append(
                {
                    "evidence_id": f"P{len(facts) + 1}",
                    "block_id": str(item["object_id"]),
                    "page_number": int(item["page_number"]),
                    "bbox": item["bbox"],
                    "quote": (
                        f"{object_type} object; {vertical_rules} vertical rules and "
                        f"{horizontal_rules} horizontal rules observed; visual content not inspected"
                    ),
                    "role": f"native_{object_type}_object",
                    "section_title": None,
                    "font_name": None,
                    "font_size_pt": None,
                    "font_flags": None,
                    "alignment": _horizontal_alignment(item["bbox"], item.get("page_width_pt")),
                    "source": item["extraction_source"],
                    "confidence": 1.0,
                    "page_rotation": 0,
                    "page_width_pt": item.get("page_width_pt"),
                    "page_height_pt": item.get("page_height_pt"),
                    "vertical_rule_count": vertical_rules,
                    "horizontal_rule_count": horizontal_rules,
                    "source_uri": f"paper://{state['task']['paper_id']}/native-object/{item['object_id']}",
                }
            )
        for item in media_rows:
            metadata = (
                item.raw_response_json.get("parser_metadata", {})
                if isinstance(item.raw_response_json, dict)
                else {}
            )
            facts.append(
                {
                    "evidence_id": f"P{len(facts) + 1}",
                    "block_id": item.object_id,
                    "page_number": item.page_number,
                    "bbox": item.bbox_json,
                    "quote": f"{item.object_type} object; visual content not inspected",
                    "role": f"native_{item.object_type}_object",
                    "section_title": None,
                    "font_name": None,
                    "font_size_pt": None,
                    "font_flags": None,
                    "alignment": None,
                    "source": str(metadata.get("extraction_source") or "pdf_object_geometry"),
                    "confidence": 1.0,
                    "page_rotation": 0,
                    "page_width_pt": metadata.get("page_width_pt"),
                    "page_height_pt": metadata.get("page_height_pt"),
                    "object_width_px": metadata.get("width_px"),
                    "object_height_px": metadata.get("height_px"),
                    "source_uri": item.source_ref,
                }
            )
        if not facts:
            raise FormatReviewFailure("FORMAT_REVIEW_NO_PAPER_EVIDENCE", "论文缺少可用于格式审查的版面产物。")
        quality = {
            "block_count": len(rows),
            "located_block_count": located,
            "native_span_count": len(native_spans),
            "native_object_count": len(media_rows),
            "native_pdf_object_count": len(native_objects),
            "layout_source": "parsed_blocks+native_pdf",
            **native_quality,
        }
        await self._emit(state, "extracting_layout", "正在提取论文版面事实")
        await self._trace(state["task"], "extract_pdf_layout", started, "succeeded", metrics=quality)
        return {"layout_facts": facts, "layout_quality": quality}

    async def build_retrieval_plan(self, state: FormatReviewState) -> dict[str, Any]:
        started = time.perf_counter()
        await self._check_cancelled(state["task_id"])
        snapshot = state["snapshot"]
        mode = str(snapshot["submission_mode"])
        manifest = _applicable_manifest(snapshot.get("rules", []), mode)
        if not manifest:
            raise FormatReviewFailure("FORMAT_RULES_UNAVAILABLE", "所选投稿模式没有可执行格式规范。")
        categories = sorted({str(item["rule_category"]) for item in manifest})
        document_ids = list(
            dict.fromkeys(
                str(value)
                for value in (snapshot.get("shared_document_id"), snapshot.get("mode_document_id"))
                if value
            )
        )
        manifest_document_ids = {
            str(item.get("source_document_id") or "") for item in manifest if item.get("source_document_id")
        }
        if not document_ids or not manifest_document_ids.issubset(set(document_ids)):
            raise FormatReviewFailure(
                "FORMAT_PROFILE_SNAPSHOT_INVALID", "格式规范与所选投稿模式的规则文档绑定不一致。"
            )
        queries = {category: _category_query(snapshot, category, manifest) for category in categories}
        plan = {
            "attempt": 0,
            "venue_id": str(snapshot.get("venue_id") or snapshot.get("profile_key")),
            "dataset_id": str(snapshot["ragflow_dataset_id"]),
            "document_ids": document_ids,
            "submission_mode": mode,
            "target_categories": categories,
            "queries": queries,
            "manifest": manifest,
            "retrieval_strategy": "ragflow_manifest_exact",
        }
        await self._emit(state, "retrieving_format_rules", "正在构建受控规范检索计划")
        await self._trace(state["task"], "build_retrieval_plan", started, "succeeded", metrics={"categories": len(categories)})
        return {"retrieval_plan": plan}

    async def retrieve_standard(self, state: FormatReviewState) -> dict[str, Any]:
        started = time.perf_counter()
        await self._check_cancelled(state["task_id"])
        await self._set_stage(state, "retrieving_format_rules", 0.40)
        plan = dict(state["retrieval_plan"])
        manifest = list(plan["manifest"])
        if plan.get("retrieval_strategy") == "ragflow_manifest_exact":
            if not self._settings.ragflow_base_url or not self._settings.ragflow_api_key:
                raise FormatReviewFailure("FORMAT_KB_UNAVAILABLE", "格式规范知识库服务尚未配置。")
            raw = await self._retrieve_manifest_chunks(plan)
            standards = _resolve_standard_chunks(raw, plan, [])
            expected_ids = {str(item["rule_id"]) for item in manifest}
            resolved_ids = {str(item["canonical_rule_id"]) for item in standards}
            missing_ids = sorted(expected_ids - resolved_ids)
            if missing_ids:
                raise FormatReviewFailure(
                    "FORMAT_KB_INCOMPLETE",
                    "格式规范知识库与规则清单不一致，无法保证完整审查。",
                    retryable=False,
                )
            rule_ids = sorted(expected_ids)
            coverage = {
                "attempt": 1,
                "strategy": "ragflow_manifest_exact",
                "covered_categories": sorted({str(item["rule_category"]) for item in manifest}),
                "missing_categories": [],
                "missing_rule_ids": [],
                "retrieved_rule_ids": rule_ids,
                "retrieved_chunk_count": len(raw),
            }
            sequence = await self._emit(
                state, "retrieving_format_rules", "已按规则清单精确装载所选投稿模式的全部规则"
            )
            await self._trace(
                state["task"],
                "retrieve_standard",
                started,
                "succeeded",
                metrics={**coverage, "standard_evidence_count": len(standards)},
            )
            return {
                "retrieval_plan": plan,
                "standard_evidences": standards,
                "coverage_report": coverage,
                "counters": {**state.get("counters", {}), "standard_retrieval": 1},
                "sequence": sequence,
            }
        if not self._settings.ragflow_base_url or not self._settings.ragflow_api_key:
            raise FormatReviewFailure("FORMAT_KB_UNAVAILABLE", "格式规范知识库服务尚未配置。")
        existing = list(state.get("standard_evidences", []))
        existing_rule_ids = {str(item.get("canonical_rule_id")) for item in existing}
        attempts = int(state.get("counters", {}).get("standard_retrieval", 0))
        while attempts < MAX_RETRIEVAL_ATTEMPTS:
            await self._check_cancelled(state["task_id"])
            missing_ids = {str(item["rule_id"]) for item in manifest} - existing_rule_ids
            if not missing_ids:
                break
            attempts += 1
            plan["attempt"] = attempts
            needed_categories = sorted(
                {str(item["rule_category"]) for item in manifest if str(item["rule_id"]) in missing_ids}
            )
            for category in needed_categories:
                await self._check_cancelled(state["task_id"])
                raw = await self._retrieve_chunks(plan, category)
                existing.extend(_resolve_standard_chunks(raw, plan, existing))
                existing_rule_ids = {str(item.get("canonical_rule_id")) for item in existing}
        if not existing:
            raise FormatReviewFailure("FORMAT_KB_NO_EVIDENCE", "未从所选格式规范库检索到规则条文。")
        required_by_category: dict[str, set[str]] = defaultdict(set)
        for item in manifest:
            required_by_category[str(item["rule_category"])].add(str(item["rule_id"]))
        returned_by_category: dict[str, set[str]] = defaultdict(set)
        for item in existing:
            returned_by_category[str(item["category"])].add(str(item["canonical_rule_id"]))
        missing_rule_ids = sorted(
            rule_id
            for category, required in required_by_category.items()
            for rule_id in required - returned_by_category[category]
        )
        missing_categories = sorted(
            category for category, required in required_by_category.items() if required - returned_by_category[category]
        )
        coverage = {
            "attempt": attempts,
            "covered_categories": sorted(set(required_by_category) - set(missing_categories)),
            "missing_categories": missing_categories,
            "missing_rule_ids": missing_rule_ids,
            "retrieved_rule_ids": sorted(existing_rule_ids),
        }
        counters = {**state.get("counters", {}), "standard_retrieval": attempts}
        sequence = await self._emit(state, "retrieving_format_rules", "正在确认规范检索覆盖范围")
        await self._trace(state["task"], "retrieve_standard", started, "succeeded", metrics=coverage)
        return {"retrieval_plan": plan, "standard_evidences": existing, "coverage_report": coverage, "counters": counters, "sequence": sequence}

    async def plan_review_units(self, state: FormatReviewState) -> dict[str, Any]:
        """Plan stable semantic/layout units from PDF facts only, never from LLM output."""

        started = time.perf_counter()
        await self._check_cancelled(state["task_id"])
        units = _plan_review_units(state["layout_facts"])
        await self._emit(
            state,
            "planning_review_units",
            f"已按论文结构规划 {len(units)} 个审查块",
        )
        await self._trace(
            state["task"],
            "plan_review_units",
            started,
            "succeeded",
            metrics={"unit_count": len(units), "unit_kinds": [unit["unit_kind"] for unit in units]},
        )
        return {"review_units": units}

    async def allocate_rules_to_units(self, state: FormatReviewState) -> dict[str, Any]:
        """Allocate complete retrieved rules by manifest scope without an LLM decision."""

        started = time.perf_counter()
        await self._check_cancelled(state["task_id"])
        units, coverage = _allocate_rules_to_units(
            units=state["review_units"],
            manifest=state["retrieval_plan"]["manifest"],
            facts=state["layout_facts"],
            standard_evidences=state["standard_evidences"],
            submission_mode=state["review"]["submission_mode"],
            retrieval_coverage=state["coverage_report"],
        )
        units, refinement_metrics = _refine_units_for_context_budget(
            units=units,
            facts=state["layout_facts"],
            standards=state["standard_evidences"],
            rules_by_id={
                str(rule["rule_id"]): rule for rule in state["retrieval_plan"]["manifest"]
            },
            submission_mode=state["review"]["submission_mode"],
            context_budget=FORMAT_MODEL_INPUT_TOKENS,
        )
        await self._persist_unit_plan(state, units)
        sequence = await self._emit(
            state,
            "allocating_rules",
            "已完成规则范围分发和块级完整性检查",
        )
        await self._trace(
            state["task"],
            "deterministic_allocate_rules_to_units",
            started,
            "succeeded",
            metrics={
                "unit_count": len(units),
                "missing_rule_ids": coverage.get("missing_rule_ids", []),
                "unallocated_rule_ids": coverage.get("unallocated_rule_ids", []),
                "context_refinement": refinement_metrics,
            },
        )
        return {"review_units": units, "coverage_report": coverage, "sequence": sequence}

    async def dispatch_units(self, state: FormatReviewState) -> dict[str, Any]:
        """Run independent units with bounded concurrency and isolated retry budgets."""

        started = time.perf_counter()
        await self._check_cancelled(state["task_id"])
        await self._set_stage(state, "reviewing_units", 0.58)
        semaphore = asyncio.Semaphore(MAX_UNIT_CONCURRENCY)
        rules_by_id = {
            str(rule["rule_id"]): rule for rule in state["retrieval_plan"]["manifest"]
        }

        async def run_one(unit: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                if not unit.get("expected_rule_ids"):
                    not_applicable_ids = {
                        str(item.get("rule_id"))
                        for item in unit.get("not_applicable_rule_ids", [])
                    }
                    unit_standards = [
                        item
                        for item in state["standard_evidences"]
                        if str(item.get("canonical_rule_id")) in not_applicable_ids
                    ]
                    findings = _not_applicable_findings(
                        unit,
                        rules_by_id,
                        unit_standards,
                        state["layout_facts"],
                    )
                    return await self._finish_unit(
                        state,
                        unit,
                        status="validated",
                        event_type="unit_validated",
                        message=(
                            "This structural block has no applicable PDF-observable rules in the selected profile."
                            if not findings
                            else "The conditional PDF-observable rules are not applicable to this structural block."
                        ),
                        findings=findings,
                        cycle_count=0,
                        retry_budget_remaining=1,
                    )
                return await self._run_review_unit(
                    state,
                    unit=unit,
                    rules_by_id=rules_by_id,
                    all_facts=state["layout_facts"],
                    all_standards=state["standard_evidences"],
                )

        raw_results = await asyncio.gather(
            *(run_one(unit) for unit in state["review_units"]), return_exceptions=True
        )
        results: list[dict[str, Any]] = []
        for unit, outcome in zip(state["review_units"], raw_results, strict=True):
            if not isinstance(outcome, Exception):
                results.append(outcome)
                continue
            reason = f"审查块执行异常：{type(outcome).__name__}。"
            findings = _unverifiable_findings_for_rules(
                list(unit.get("expected_rule_ids", [])),
                rules_by_id,
                reason,
                facts=_facts_for_unit(unit, state["layout_facts"]),
                standards=state["standard_evidences"],
            )
            try:
                results.append(
                    await self._finish_unit(
                        state,
                        unit,
                        status="failed",
                        event_type="unit_failed",
                        message="该审查块失败，其他块将继续完成并参与汇总。",
                        findings=findings,
                        cycle_count=int(unit.get("unit_cycle_count") or 0),
                        retry_budget_remaining=int(unit.get("retry_budget_remaining") or 0),
                        last_retry_reason=reason,
                    )
                )
            except Exception as persist_error:
                # A database/event failure must still not cancel sibling units.
                # The final report will retain this unit as an explicit gap.
                results.append(
                    {
                        "unit_id": unit["unit_id"],
                        "unit_position": unit["unit_position"],
                        "status": "failed",
                        "findings": findings,
                        "coverage": unit.get("coverage", {}),
                        "persistence_error": type(persist_error).__name__,
                    }
                )
        async with self._sessions() as session:
            persisted_units = list(
                (
                    await session.scalars(
                        select(FormatReviewUnit).where(
                            FormatReviewUnit.format_review_id == state["review_id"]
                        )
                    )
                ).all()
            )
        terminal_statuses = {"validated", "unverifiable", "failed", "cancelled", "skipped"}
        non_terminal_ids = sorted(
            row.unit_id for row in persisted_units if row.status not in terminal_statuses
        )
        persisted_ids = {row.unit_id for row in persisted_units}
        missing_ids = sorted(
            str(unit["unit_id"])
            for unit in state["review_units"]
            if str(unit["unit_id"]) not in persisted_ids
        )
        if non_terminal_ids or missing_ids:
            raise FormatReviewFailure(
                "FORMAT_REVIEW_UNIT_PERSISTENCE_FAILED",
                "审查块终态未完整持久化，禁止生成汇总结果。",
                retryable=True,
            )
        sequence = await self._emit(state, "reviewing_units", "全部审查块已达到确定终态")
        await self._trace(
            state["task"],
            "dispatch_units",
            started,
            "succeeded",
            metrics={
                "unit_count": len(results),
                "max_concurrency": MAX_UNIT_CONCURRENCY,
                "statuses": {item["unit_id"]: item["status"] for item in results},
            },
        )
        return {"unit_results": results, "sequence": sequence}

    async def synthesize_units(self, state: FormatReviewState) -> dict[str, Any]:
        """Synthesize only persisted, validated block findings; never inspect the PDF again."""

        started = time.perf_counter()
        await self._check_cancelled(state["task_id"])
        await self._emit_synthesis_event(state, "synthesis_started", "正在汇总已核验审查块结果")
        findings = _deduplicate_unit_findings(state.get("unit_results", []))
        deterministic_summary = _summary_markdown(findings, state["coverage_report"])
        summary = deterministic_summary
        if findings and self._settings.llm_api_key:
            runner = AgentRunner(self._settings, state["snapshot"].get("configuration"))
            try:
                synthesis_payload = _synthesis_payload(findings, state["coverage_report"])
                if (
                    format_review_input_tokens("format_synthesis", synthesis_payload)
                    <= FORMAT_MODEL_INPUT_TOKENS
                ):
                    output, metrics = await runner.invoke(
                        "format_synthesis", FormatSynthesisOutput, payload=synthesis_payload
                    )
                    summary = output.summary_markdown
                    state.setdefault("metrics", {})["format_synthesis"] = metrics
            except Exception:
                # The final gate remains deterministic; a summary-model failure
                # must not discard already validated user-visible findings.
                summary = deterministic_summary
        await self._emit_synthesis_event(state, "synthesis_completed", "已完成结构化结果汇总")
        await self._trace(
            state["task"],
            "FormatSynthesisAgent",
            started,
            "succeeded",
            metrics={"input_finding_count": sum(len(item.get("findings", [])) for item in state.get("unit_results", [])), "deduplicated_finding_count": len(findings)},
        )
        return {"final_findings": findings, "summary_markdown": summary}

    async def check_format(self, state: FormatReviewState) -> dict[str, Any]:
        started = time.perf_counter()
        await self._check_cancelled(state["task_id"])
        await self._set_stage(state, "checking_rules", 0.62)
        plan = state["retrieval_plan"]
        standards = state["standard_evidences"]
        facts = state["layout_facts"]
        coverage = state["coverage_report"]
        runner = AgentRunner(self._settings, state["snapshot"].get("configuration"))
        candidate_payloads: list[dict[str, Any]] = []
        metrics: dict[str, Any] = dict(state.get("metrics", {}))
        grouped = _context_groups(plan, standards, facts)
        for category, payload in grouped:
            if payload is None:
                coverage["missing_categories"] = sorted(
                    set(coverage.get("missing_categories", [])) | {category}
                )
                continue
            if format_review_input_tokens("format_check", payload) > FORMAT_MODEL_INPUT_TOKENS:
                coverage["missing_categories"] = sorted(set(coverage.get("missing_categories", [])) | {category})
                continue
            try:
                output, call_metrics = await runner.invoke("format_check", CompositeFormatReviewOutput, payload=payload)
            except Exception as exc:
                raise FormatReviewFailure("FORMAT_CHECK_FAILED", "格式综合检查模型调用失败。", retryable=True) from exc
            metrics[f"check_format:{category}"] = call_metrics
            allowed_categories = set(payload["target_categories"])
            candidate_payloads.extend(
                item.model_dump(mode="json")
                for item in output.findings
                if item.category in allowed_categories
            )
            if output.summary_markdown and not state.get("summary_markdown"):
                state["summary_markdown"] = output.summary_markdown
        valid = validate_findings(
            candidate_payloads,
            paper_evidences=facts,
            standard_evidences=standards,
            coverage_report=coverage,
        )
        represented_categories = {item["category"] for item in valid}
        for category in plan["target_categories"]:
            if category not in represented_categories:
                valid.append(
                    {
                        "category": category,
                        "aspect": f"{category} 类格式要求",
                        "result": "unverifiable",
                        "severity": "info",
                        "finding": "没有得到可由证据支持的格式结论。",
                        "suggestion": None,
                        "paper_evidences": [],
                        "standard_evidences": [],
                        "evidence_status": "incomplete",
                        "reason": "模型输出未形成可验证发现。",
                    }
                )
        valid.extend(
            item for item in forced_unverifiable_findings(coverage) if item["category"] not in represented_categories
        )
        sequence = await self._emit(state, "checking_rules", "正在进行综合格式检查")
        await self._trace(state["task"], "check_format", started, "succeeded", metrics=metrics)
        return {"candidates": valid, "metrics": metrics, "summary_markdown": state.get("summary_markdown", ""), "sequence": sequence}

    async def reflect_validation(self, state: FormatReviewState) -> dict[str, Any]:
        started = time.perf_counter()
        await self._check_cancelled(state["task_id"])
        await self._set_stage(state, "validating_evidence", 0.78)
        candidates = list(state.get("candidates", []))
        counters = dict(state.get("counters", {}))
        actionable = [item for item in candidates if item.get("result") in {"compliant", "non_compliant"}]
        if not actionable:
            route, reason = "confirmed", "所有项目均已按证据门禁标记为不可判定或不适用。"
        else:
            runner = AgentRunner(self._settings, state["snapshot"].get("configuration"))
            try:
                reflected, call_metrics = await runner.invoke(
                    "format_reflect",
                    ReflectionOutput,
                    payload={"findings": candidates, "coverage_report": state["coverage_report"]},
                )
            except Exception as exc:
                raise FormatReviewFailure("FORMAT_VALIDATION_FAILED", "格式证据核验模型调用失败。", retryable=True) from exc
            state.setdefault("metrics", {})["reflect_validation"] = call_metrics
            route, reason = reflected.decision, reflected.reason
        if route == "recover_pdf_evidence":
            attempts = counters.get("pdf_evidence_recovery", 0)
            if attempts >= MAX_EVIDENCE_RECOVERY:
                route = "confirmed"
                candidates = _make_unverifiable(candidates, "论文版面证据恢复达到上限。")
        elif route in {"retrieve_standard", "clarify_standard"}:
            attempts = counters.get("standard_retrieval", 0)
            if attempts >= MAX_RETRIEVAL_ATTEMPTS:
                route = "confirmed"
                candidates = _make_unverifiable(candidates, "规范澄清检索达到上限。")
        elif route == "repair_check":
            attempts = counters.get("validation_repair", 0)
            if attempts >= MAX_VALIDATION_REPAIRS:
                route = "confirmed"
                candidates = _make_unverifiable(candidates, "格式检查修复达到上限。")
            else:
                counters["validation_repair"] = attempts + 1
        elif route == "unverifiable":
            route = "confirmed"
            candidates = _make_unverifiable(candidates, reason)
        sequence = await self._emit(state, "validating_evidence", "正在核验规范与论文双证据")
        await self._trace(state["task"], "reflect_validation", started, "succeeded", metrics={"route": route})
        return {
            "candidates": candidates,
            "route": route,
            "counters": counters,
            "metrics": state.get("metrics", {}),
            "sequence": sequence,
        }

    async def recover_pdf_evidence(self, state: FormatReviewState) -> dict[str, Any]:
        started = time.perf_counter()
        await self._check_cancelled(state["task_id"])
        counters = dict(state.get("counters", {}))
        counters["pdf_evidence_recovery"] = counters.get("pdf_evidence_recovery", 0) + 1
        # Parsed layout records are the only trusted local fact source. A
        # recovery pass reuses every stored block rather than asking an LLM to
        # manufacture font sizes or coordinates.
        sequence = await self._emit(state, "extracting_layout", "正在恢复缺失的论文版面证据")
        await self._trace(state["task"], "recover_pdf_evidence", started, "succeeded", metrics=counters)
        return {"counters": counters, "sequence": sequence}

    async def generate_report(self, state: FormatReviewState) -> dict[str, Any]:
        started = time.perf_counter()
        await self._check_cancelled(state["task_id"])
        await self._set_stage(state, "generating_annotations", 0.92)
        findings = list(state.get("final_findings", []))
        summary = state.get("summary_markdown") or _summary_markdown(findings, state["coverage_report"])
        annotations = _annotations(findings)
        async with self._sessions() as session:
            task = await session.get(TaskRecord, state["task_id"])
            review = await session.get(FormatReview, state["review_id"])
            if task is None or review is None:
                raise FormatReviewFailure("FORMAT_REVIEW_NOT_FOUND", "格式审查任务不存在。")
            await session.execute(delete(FormatReviewItem).where(FormatReviewItem.format_review_id == review.format_review_id))
            records = [
                _item_record(
                    review.format_review_id,
                    finding,
                    unit_id=finding.get("unit_id"),
                    unit_position=finding.get("unit_position"),
                    source_stage="final",
                )
                for finding in findings
            ]
            session.add_all(records)
            now = datetime.now(UTC)
            review.status = "succeeded"
            review.summary_markdown = summary
            review.coverage_report_json = state["coverage_report"]
            review.annotation_json = {"items": annotations, "coordinate_space": "page_top_left_pt"}
            review.metrics_json = state.get("metrics", {})
            review.synthesis_status = "completed"
            review.error_json = None
            review.completed_at = now
            task.status, task.stage, task.progress, task.completed_at = "succeeded", "completed", 1.0, now
            task.result_json = {
                "format_review_id": review.format_review_id,
                "finding_count": len(records),
                "unverifiable_count": sum(item.result == "unverifiable" for item in records),
            }
            await session.commit()
            task_state = task_view(task)
        await self._redis.set_task_state(state["task_id"], task_state)
        sequence = await self._emit(state, "completed", "格式审查报告已生成", terminal=True)
        await self._trace(state["task"], "generate_report", started, "succeeded", metrics={"findings": len(findings)})
        return {"findings": findings, "summary_markdown": summary, "sequence": sequence}

    async def _retrieve_chunks(self, plan: dict[str, Any], category: str) -> list[dict[str, Any]]:
        payload = {
            "question": plan["queries"][category],
            "dataset_ids": [plan["dataset_id"]],
            "document_ids": plan["document_ids"],
            "top_k": 24,
        }
        headers = {"Authorization": f"Bearer {self._settings.ragflow_api_key.get_secret_value()}"}
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
                    response = await client.post(
                        _ragflow_endpoint(self._settings.ragflow_base_url, "retrieval"),
                        json=payload,
                        headers=headers,
                    )
                    response.raise_for_status()
                    body = response.json()
                break
            except (httpx.HTTPError, ValueError) as exc:
                if attempt == 0:
                    await asyncio.sleep(0)
                    continue
                raise FormatReviewFailure(
                    "FORMAT_KB_UNAVAILABLE", "格式规范知识库检索失败。", retryable=True
                ) from exc
        data = body.get("data", body) if isinstance(body, dict) else {}
        raw = data.get("chunks", data.get("items", [])) if isinstance(data, dict) else []
        return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []

    async def _retrieve_manifest_chunks(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        """Retrieve every atomic chunk in each selected document, independent of similarity rank."""

        headers = {"Authorization": f"Bearer {self._settings.ragflow_api_key.get_secret_value()}"}
        collected: list[dict[str, Any]] = []
        manifest = list(plan["manifest"])
        for document_id in plan["document_ids"]:
            expected = [
                item for item in manifest if str(item.get("source_document_id")) == document_id
            ]
            if len(expected) > 100:
                raise FormatReviewFailure(
                    "FORMAT_PROFILE_SNAPSHOT_INVALID",
                    "单个规则文档超过知识库全量检索上限，请拆分规则文档。",
                )
            payload = {
                "question": (
                    f"{plan['venue_id']} {plan['submission_mode']} all atomic manuscript "
                    "format rules page layout title heading abstract author figure table "
                    "reference appendix"
                ),
                "dataset_ids": [plan["dataset_id"]],
                "document_ids": [document_id],
                "top_k": 100,
                "page_size": 100,
                "similarity_threshold": 0,
            }
            for attempt in range(2):
                try:
                    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
                        response = await client.post(
                            _ragflow_endpoint(self._settings.ragflow_base_url, "retrieval"),
                            json=payload,
                            headers=headers,
                        )
                        response.raise_for_status()
                        body = response.json()
                    if isinstance(body, dict) and body.get("code") not in {None, 0}:
                        raise ValueError(str(body.get("message") or "RAGFlow retrieval failed"))
                    break
                except (httpx.HTTPError, ValueError) as exc:
                    if attempt == 0:
                        await asyncio.sleep(0)
                        continue
                    raise FormatReviewFailure(
                        "FORMAT_KB_UNAVAILABLE",
                        "格式规范知识库检索失败。",
                        retryable=True,
                    ) from exc
            data = body.get("data", body) if isinstance(body, dict) else {}
            raw = data.get("chunks", data.get("items", [])) if isinstance(data, dict) else []
            if isinstance(raw, list):
                collected.extend(item for item in raw if isinstance(item, dict))
        return collected

    async def _load_native_spans(
        self,
        *,
        paper_id: str,
        paper_version_id: str,
        file_path: str,
    ) -> tuple[list[PdfTextSpanRecord], list[dict[str, Any]], dict[str, Any]]:
        async with self._sessions() as session:
            existing = list(
                (
                    await session.scalars(
                        select(PdfTextSpanRecord)
                        .where(
                            PdfTextSpanRecord.paper_id == paper_id,
                            PdfTextSpanRecord.paper_version_id == paper_version_id,
                        )
                        .order_by(PdfTextSpanRecord.span_index)
                    )
                ).all()
            )
        layout = extract_native_pdf_layout(file_path)
        if existing:
            return existing, layout.objects, {
                "native_pdf_available": layout.available,
                "native_pdf_cached": True,
                "native_pdf_object_count": len(layout.objects),
                "native_pdf_reason": layout.reason,
            }
        if not layout.spans:
            return [], layout.objects, {
                "native_pdf_available": layout.available,
                "native_pdf_cached": False,
                "native_pdf_reason": layout.reason,
                "page_count": layout.page_count,
                "native_pdf_object_count": len(layout.objects),
            }
        records = [
            PdfTextSpanRecord(
                paper_id=paper_id,
                paper_version_id=paper_version_id,
                span_index=index,
                **_pdf_span_record_values(span),
            )
            for index, span in enumerate(layout.spans)
        ]
        async with self._sessions() as session:
            session.add_all(records)
            try:
                await session.commit()
            except IntegrityError:
                # A concurrent review may have populated the immutable cache
                # after our initial lookup. Reuse those facts rather than fail.
                await session.rollback()
                records = list(
                    (
                        await session.scalars(
                            select(PdfTextSpanRecord)
                            .where(
                                PdfTextSpanRecord.paper_id == paper_id,
                                PdfTextSpanRecord.paper_version_id == paper_version_id,
                            )
                            .order_by(PdfTextSpanRecord.span_index)
                        )
                    ).all()
                )
        return records, layout.objects, {
            "native_pdf_available": True,
            "native_pdf_cached": False,
            "page_count": layout.page_count,
            "native_pdf_object_count": len(layout.objects),
        }

    async def _persist_unit_plan(self, state: FormatReviewState, units: list[dict[str, Any]]) -> None:
        """Persist plan/allocation before any unit event makes it visible to a client."""

        async with self._sessions() as session:
            review = await session.get(FormatReview, state["review_id"])
            if review is None:
                raise FormatReviewFailure("FORMAT_REVIEW_NOT_FOUND", "格式审查任务不存在。")
            await session.execute(
                delete(FormatReviewUnit).where(FormatReviewUnit.format_review_id == review.format_review_id)
            )
            records = [
                FormatReviewUnit(
                    format_review_id=review.format_review_id,
                    unit_id=str(unit["unit_id"]),
                    unit_position=int(unit["unit_position"]),
                    unit_kind=str(unit["unit_kind"]),
                    title=str(unit["title"]),
                    page_range_json=list(unit.get("page_range", [])),
                    block_ids_json=list(unit.get("block_ids", [])),
                    expected_rule_ids_json=list(unit.get("expected_rule_ids", [])),
                    allocated_rule_ids_json=list(unit.get("allocated_rule_ids", [])),
                    global_rule_ids_json=list(unit.get("global_rule_ids", [])),
                    not_applicable_rule_ids_json=list(unit.get("not_applicable_rule_ids", [])),
                    retrieved_rule_ids_json=list(unit.get("retrieved_rule_ids", [])),
                    coverage_json=dict(unit.get("coverage", {})),
                    status="pending",
                    unit_cycle_count=0,
                    retry_budget_remaining=1,
                    event_sequence=0,
                )
                for unit in units
            ]
            session.add_all(records)
            review.unit_plan_json = [_public_unit_plan(unit) for unit in units]
            review.synthesis_status = "pending"
            await session.commit()

    async def _run_review_unit(
        self,
        state: FormatReviewState,
        *,
        unit: dict[str, Any],
        rules_by_id: dict[str, dict[str, Any]],
        all_facts: list[dict[str, Any]],
        all_standards: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run one unit through at most two full check/validation cycles."""

        await self._check_cancelled(state["task_id"])
        facts = _facts_for_unit(unit, all_facts)
        expected_rule_ids = list(unit.get("expected_rule_ids", []))
        not_applicable_rule_ids = {
            str(item.get("rule_id")) for item in unit.get("not_applicable_rule_ids", [])
        }
        standards = [
            item
            for item in all_standards
            if str(item.get("canonical_rule_id")) in set(expected_rule_ids) | not_applicable_rule_ids
        ]
        retrieved_rule_ids = {str(item.get("canonical_rule_id")) for item in standards}
        missing_rule_ids = sorted(set(expected_rule_ids) - retrieved_rule_ids)
        executable_rule_ids = sorted(set(expected_rule_ids) & retrieved_rule_ids)
        coverage = {
            **dict(unit.get("coverage", {})),
            "expected_rule_ids": expected_rule_ids,
            "retrieved_rule_ids": sorted(retrieved_rule_ids),
            "missing_rule_ids": missing_rule_ids,
            "missing_categories": sorted(
                {
                    str(rules_by_id[rule_id]["rule_category"])
                    for rule_id in missing_rule_ids
                    if rule_id in rules_by_id
                }
            ),
        }
        unit["coverage"] = coverage
        unit["retrieved_rule_ids"] = sorted(retrieved_rule_ids)
        await self._persist_unit_event(
            state,
            unit,
            event_type="unit_started",
            status="running",
            message="审查块已开始，正在装配完整规则与论文事实。",
            cycle_count=0,
            retry_budget_remaining=1,
        )

        not_applicable = _not_applicable_findings(unit, rules_by_id, standards, all_facts)
        missing_findings = _unverifiable_findings_for_rules(
            missing_rule_ids,
            rules_by_id,
            "适用规范未完整检索，无法可靠判断。",
            facts=facts,
            standards=standards,
        )
        retained_findings = not_applicable + missing_findings
        if not executable_rule_ids:
            return await self._finish_unit(
                state,
                unit,
                status="unverifiable",
                event_type="unit_unverifiable",
                message="审查块没有可执行的已检索规则，已标记为无法可靠判断。",
                findings=retained_findings,
                cycle_count=0,
                retry_budget_remaining=1,
            )

        execution_unit = {
            **unit,
            "expected_rule_ids": executable_rule_ids,
            "retrieved_rule_ids": executable_rule_ids,
        }

        runner = AgentRunner(self._settings, state["snapshot"].get("configuration"))
        cycle_count = 0
        retry_budget_remaining = 1
        last_retry_reason: str | None = None
        while cycle_count < MAX_UNIT_CYCLES:
            await self._check_cancelled(state["task_id"])
            cycle_count += 1
            await self._persist_unit_event(
                state,
                unit,
                event_type="unit_progress",
                status="running",
                message=f"正在执行第 {cycle_count} 次块级检查与证据核验。",
                cycle_count=cycle_count,
                retry_budget_remaining=retry_budget_remaining,
                last_retry_reason=last_retry_reason,
            )
            payload = _unit_context_payload(
                unit=execution_unit,
                facts=facts,
                standards=standards,
                rules_by_id=rules_by_id,
                submission_mode=state["review"]["submission_mode"],
            )
            if format_review_input_tokens("format_check", payload) > FORMAT_MODEL_INPUT_TOKENS:
                findings = retained_findings + _unverifiable_findings_for_rules(
                    executable_rule_ids,
                    rules_by_id,
                    "该审查块无法在不截断完整规则的前提下装配到模型上下文。",
                    facts=facts,
                    standards=standards,
                )
                return await self._finish_unit(
                    state,
                    unit,
                    status="unverifiable",
                    event_type="unit_unverifiable",
                    message="审查块上下文超出安全预算，未作合规推断。",
                    findings=findings,
                    cycle_count=cycle_count,
                    retry_budget_remaining=retry_budget_remaining,
                    last_retry_reason=last_retry_reason,
                )
            try:
                output, check_metrics = await runner.invoke(
                    "format_check", CompositeFormatReviewOutput, payload=payload
                )
                candidates = [
                    item.model_dump(mode="json")
                    for item in output.findings
                    if item.category in set(payload["target_categories"])
                ]
                validated = validate_findings(
                    candidates,
                    paper_evidences=facts,
                    standard_evidences=standards,
                    coverage_report=coverage,
                )
                validated.extend(retained_findings)
                validated = _ensure_unit_rule_representation(
                    validated,
                    expected_rule_ids,
                    rules_by_id,
                    facts=facts,
                    standards=standards,
                )
                actionable = [
                    item for item in validated if item.get("result") in {"compliant", "non_compliant"}
                ]
                if actionable:
                    reflected, validation_metrics = await runner.invoke(
                        "format_reflect",
                        ReflectionOutput,
                        payload=_reflection_payload(unit, validated, coverage),
                    )
                    decision, reason = reflected.decision, reflected.reason
                else:
                    validation_metrics = {}
                    decision, reason = "confirmed", "全部候选已由确定性证据门禁处理。"
            except Exception as exc:
                decision, reason = "repair_check", "模型调用或结构化校验失败。"
                validated = retained_findings + _unverifiable_findings_for_rules(
                    executable_rule_ids,
                    rules_by_id,
                    "块级模型调用失败，无法可靠判断。",
                    facts=facts,
                    standards=standards,
                )
                check_metrics, validation_metrics = {"error": type(exc).__name__}, {}

            state.setdefault("metrics", {})[
                f"unit:{unit['unit_id']}:cycle:{cycle_count}"
            ] = {"check": check_metrics, "validation": validation_metrics, "decision": decision}
            if check_metrics.get("error") and cycle_count >= MAX_UNIT_CYCLES:
                return await self._finish_unit(
                    state,
                    unit,
                    status="failed",
                    event_type="unit_failed",
                    message="The model did not return a valid response after the bounded retry; no format conclusion was produced.",
                    findings=validated,
                    cycle_count=cycle_count,
                    retry_budget_remaining=0,
                    last_retry_reason=f"Model call failed: {check_metrics['error']}",
                )
            if decision == "confirmed":
                status = (
                    "validated"
                    if any(item.get("result") in {"compliant", "non_compliant", "not_applicable"} for item in validated)
                    else "unverifiable"
                )
                return await self._finish_unit(
                    state,
                    unit,
                    status=status,
                    event_type="unit_validated" if status == "validated" else "unit_unverifiable",
                    message="审查块已完成证据核验。" if status == "validated" else "审查块没有可验证结论。",
                    findings=validated,
                    cycle_count=cycle_count,
                    retry_budget_remaining=retry_budget_remaining,
                    last_retry_reason=last_retry_reason,
                )
            unresolved_rule_ids = _unresolved_unit_rule_ids(validated, expected_rule_ids)
            if decision in {"recover_pdf_evidence", "retrieve_standard", "clarify_standard"}:
                if not unresolved_rule_ids:
                    return await self._finish_unit(
                        state,
                        unit,
                        status="validated",
                        event_type="unit_validated",
                        message="该审查块的全部原子规则均已有完整可核验结论。",
                        findings=validated,
                        cycle_count=cycle_count,
                        retry_budget_remaining=retry_budget_remaining,
                        last_retry_reason=reason,
                    )
                if cycle_count >= MAX_UNIT_CYCLES:
                    return await self._finish_unit(
                        state,
                        unit,
                        status="validated" if _has_complete_actionable_finding(validated) else "unverifiable",
                        event_type="unit_validated" if _has_complete_actionable_finding(validated) else "unit_unverifiable",
                        message="补正预算已用尽；未获得新增可核验证据的原子规则已标记为无法可靠判断。",
                        findings=_force_rules_unverifiable(validated, unresolved_rule_ids, reason),
                        cycle_count=cycle_count,
                        retry_budget_remaining=0,
                        last_retry_reason=reason,
                    )
                if decision == "recover_pdf_evidence":
                    try:
                        supplements = await asyncio.to_thread(
                            supplement_unit_evidence,
                            pdf_path=state["task"]["paper_file_path"],
                            unit=unit,
                            rules_by_id=rules_by_id,
                            unresolved_rule_ids=unresolved_rule_ids,
                            existing_facts=all_facts,
                        )
                    except Exception:
                        # Evidence recovery must not turn a local uncertainty
                        # into a failed review unit.
                        supplements = []
                    supplements = _new_supplemental_facts(all_facts, supplements)
                    if supplements:
                        all_facts = [*all_facts, *supplements]
                        facts = _facts_for_unit(unit, all_facts)
                    recovery_detail = "PDF 补正器未返回新增可定位事实。"
                    has_recovery_delta = bool(supplements)
                else:
                    refreshed = self._refresh_standard_evidences_for_rules(
                        state, unresolved_rule_ids
                    )
                    refreshed = _new_standard_evidences(all_standards, refreshed)
                    if refreshed:
                        all_standards = [
                            item
                            for item in all_standards
                            if str(item.get("canonical_rule_id")) not in set(unresolved_rule_ids)
                        ] + refreshed
                        standards = [
                            item
                            for item in all_standards
                            if str(item.get("canonical_rule_id"))
                            in set(unresolved_rule_ids) | not_applicable_rule_ids
                        ]
                        executable_rule_ids = sorted(
                            set(unresolved_rule_ids)
                            & {str(item.get("canonical_rule_id")) for item in standards}
                        )
                    recovery_detail = "冻结规则清单未提供可用的精确标准补充。"
                    has_recovery_delta = bool(refreshed)
                if not has_recovery_delta:
                    return await self._finish_unit(
                        state,
                        unit,
                        status="validated" if _has_complete_actionable_finding(validated) else "unverifiable",
                        event_type="unit_validated" if _has_complete_actionable_finding(validated) else "unit_unverifiable",
                        message="补正未产生增量事实；相关原子规则已标记为无法可靠判断。",
                        findings=_force_rules_unverifiable(
                            validated, unresolved_rule_ids, f"{reason} {recovery_detail}"
                        ),
                        cycle_count=cycle_count,
                        retry_budget_remaining=retry_budget_remaining,
                        last_retry_reason=f"{reason} {recovery_detail}",
                    )
                retained_findings = _freeze_complete_findings(validated, unresolved_rule_ids)
                execution_unit = {
                    **unit,
                    "expected_rule_ids": executable_rule_ids,
                    "retrieved_rule_ids": executable_rule_ids,
                }
                retry_budget_remaining = 0
                last_retry_reason = reason
                await self._persist_unit_event(
                    state,
                    unit,
                    event_type="unit_progress",
                    status="running",
                    message="已补充原子规则的可追溯证据，正在仅重审未解决规则。",
                    cycle_count=cycle_count,
                    retry_budget_remaining=retry_budget_remaining,
                    last_retry_reason=last_retry_reason,
                )
                continue
            # Preserve independently complete conclusions when reflection does
            # not identify a recoverable evidence path.
            if _has_complete_actionable_finding(validated):
                return await self._finish_unit(
                    state,
                    unit,
                    status="validated",
                    event_type="unit_validated",
                    message="审查块保留已完整核验的结论；其余规则因缺少局部锚点单独标记为无法可靠判断。",
                    findings=validated,
                    cycle_count=cycle_count,
                    retry_budget_remaining=retry_budget_remaining,
                    last_retry_reason=reason,
                )
            if decision == "unverifiable":
                return await self._finish_unit(
                    state,
                    unit,
                    status="unverifiable",
                    event_type="unit_unverifiable",
                    message="证据核验要求将该审查块标记为无法可靠判断。",
                    findings=_force_rules_unverifiable(validated, unresolved_rule_ids, reason),
                    cycle_count=cycle_count,
                    retry_budget_remaining=retry_budget_remaining,
                    last_retry_reason=reason,
                )

            # repair_check uses one same-context retry for model or schema
            # instability. Evidence recovery is handled above with a concrete
            # supplement before any second model call.
            if cycle_count < MAX_UNIT_CYCLES:
                retry_budget_remaining = 0
                last_retry_reason = reason
                await self._persist_unit_event(
                    state,
                    unit,
                    event_type="unit_progress",
                    status="running",
                    message="证据核验请求一次完整重试。",
                    cycle_count=cycle_count,
                    retry_budget_remaining=retry_budget_remaining,
                    last_retry_reason=last_retry_reason,
                )
                continue
            return await self._finish_unit(
                state,
                unit,
                status="unverifiable",
                event_type="unit_unverifiable",
                message="审查块已用尽一次完整重试预算。",
                findings=_force_rules_unverifiable(validated, unresolved_rule_ids, reason),
                cycle_count=cycle_count,
                retry_budget_remaining=0,
                last_retry_reason=reason,
            )

        # Defensive terminal path; the loop is bounded but all paths must be explicit.
        return await self._finish_unit(
            state,
            unit,
            status="failed",
            event_type="unit_failed",
            message="审查块未能达到确定终态。",
            findings=_unverifiable_findings_for_rules(
                    expected_rule_ids,
                    rules_by_id,
                    "审查块执行异常，无法可靠判断。",
                    facts=facts,
                    standards=standards,
                ),
            cycle_count=cycle_count,
            retry_budget_remaining=0,
            last_retry_reason=last_retry_reason,
        )

    def _refresh_standard_evidences_for_rules(
        self, state: FormatReviewState, rule_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Rehydrate frozen atomic rule text without a similarity search."""

        plan = state["retrieval_plan"]
        requested = set(rule_ids)
        manifest = [
            item for item in plan["manifest"] if str(item.get("rule_id")) in requested
        ]
        evidences = _manifest_standard_evidences(manifest, list(plan["document_ids"]))
        for evidence in evidences:
            rule_id = str(evidence["canonical_rule_id"])
            evidence["evidence_id"] = f"S-REF-{hashlib.sha256(rule_id.encode()).hexdigest()[:12]}"
            evidence["retrieval_strategy"] = "manifest_refresh_exact"
        return evidences

    async def _finish_unit(
        self,
        state: FormatReviewState,
        unit: dict[str, Any],
        *,
        status: str,
        event_type: str,
        message: str,
        findings: list[dict[str, Any]],
        cycle_count: int,
        retry_budget_remaining: int,
        last_retry_reason: str | None = None,
    ) -> dict[str, Any]:
        await self._persist_unit_event(
            state,
            unit,
            event_type=event_type,
            status=status,
            message=message,
            findings=findings,
            cycle_count=cycle_count,
            retry_budget_remaining=retry_budget_remaining,
            last_retry_reason=last_retry_reason,
        )
        return {
            "unit_id": unit["unit_id"],
            "unit_position": unit["unit_position"],
            "status": status,
            "findings": findings,
            "coverage": unit["coverage"],
        }

    async def _persist_unit_event(
        self,
        state: FormatReviewState,
        unit: dict[str, Any],
        *,
        event_type: str,
        status: str,
        message: str,
        cycle_count: int,
        retry_budget_remaining: int,
        last_retry_reason: str | None = None,
        findings: list[dict[str, Any]] | None = None,
    ) -> int:
        """Commit unit state/findings first, then append its resumable SSE event."""

        async with self._sessions() as session:
            review = await session.scalar(
                select(FormatReview)
                .where(FormatReview.format_review_id == state["review_id"])
                .with_for_update()
            )
            row = await session.scalar(
                select(FormatReviewUnit).where(
                    FormatReviewUnit.format_review_id == state["review_id"],
                    FormatReviewUnit.unit_id == unit["unit_id"],
                )
            )
            if review is None or row is None:
                raise FormatReviewFailure("FORMAT_REVIEW_UNIT_NOT_FOUND", "审查块状态不存在。")
            now = datetime.now(UTC)
            row.status = status
            row.unit_cycle_count = cycle_count
            row.retry_budget_remaining = retry_budget_remaining
            row.last_retry_reason = last_retry_reason
            row.coverage_json = dict(unit.get("coverage", {}))
            row.retrieved_rule_ids_json = list(unit.get("retrieved_rule_ids", []))
            row.started_at = row.started_at or now
            if status in {"validated", "unverifiable", "failed", "cancelled", "skipped"}:
                row.completed_at = now
            if findings is not None:
                row.validated_findings_json = findings
                await session.execute(
                    delete(FormatReviewItem).where(
                        FormatReviewItem.format_review_id == review.format_review_id,
                        FormatReviewItem.unit_id == unit["unit_id"],
                        FormatReviewItem.source_stage == "unit",
                    )
                )
                session.add_all(
                    _item_record(
                        review.format_review_id,
                        finding,
                        unit_id=unit["unit_id"],
                        unit_position=unit["unit_position"],
                        source_stage="unit",
                    )
                    for finding in findings
                )
            review.event_sequence += 1
            row.event_sequence = review.event_sequence
            await session.commit()
            sequence = review.event_sequence
        unit.update(
            {
                "status": status,
                "unit_cycle_count": cycle_count,
                "retry_budget_remaining": retry_budget_remaining,
                "last_retry_reason": last_retry_reason,
                "event_sequence": sequence,
            }
        )
        data: dict[str, Any] = {
            "unit_id": unit["unit_id"],
            "unit_position": unit["unit_position"],
            "unit_kind": unit["unit_kind"],
            "title": unit["title"],
            "page_range": unit.get("page_range", []),
            "status": status,
            "unit_cycle_count": cycle_count,
            "retry_budget_remaining": retry_budget_remaining,
            "last_retry_reason": last_retry_reason,
            "event_sequence": sequence,
            "coverage": unit.get("coverage", {}),
            "message": message,
        }
        if findings is not None:
            data["findings"] = findings
            data["evidence_ids"] = sorted(
                {
                    str(evidence.get("evidence_id"))
                    for finding in findings
                    for evidence in [*finding.get("paper_evidences", []), *finding.get("standard_evidences", [])]
                    if evidence.get("evidence_id")
                }
            )
        await self._redis.append_event(
            StreamEvent(
                event_id=str(uuid4()),
                event_type=event_type,  # type: ignore[arg-type]
                task_id=state["task_id"],
                message_id=state["review_id"],
                session_id=state["review_id"],
                sequence=sequence,
                timestamp=datetime.now(UTC),
                data=data,
            )
        )
        return sequence

    async def _emit_synthesis_event(self, state: FormatReviewState, event_type: str, message: str) -> int:
        async with self._sessions() as session:
            review = await session.scalar(
                select(FormatReview)
                .where(FormatReview.format_review_id == state["review_id"])
                .with_for_update()
            )
            if review is None:
                raise FormatReviewFailure("FORMAT_REVIEW_NOT_FOUND", "格式审查任务不存在。")
            review.synthesis_status = "running" if event_type == "synthesis_started" else "completed"
            review.event_sequence += 1
            await session.commit()
            sequence = review.event_sequence
        await self._redis.append_event(
            StreamEvent(
                event_id=str(uuid4()),
                event_type=event_type,  # type: ignore[arg-type]
                task_id=state["task_id"],
                message_id=state["review_id"],
                session_id=state["review_id"],
                sequence=sequence,
                timestamp=datetime.now(UTC),
                data={"event_sequence": sequence, "status": "running" if event_type == "synthesis_started" else "completed", "message": message},
            )
        )
        return sequence

    async def _set_stage(self, state: FormatReviewState, stage: str, progress: float) -> None:
        async with self._sessions() as session:
            task = await session.get(TaskRecord, state["task_id"])
            if task is None:
                raise FormatReviewFailure("TASK_NOT_FOUND", "格式审查任务不存在。")
            task.stage, task.progress = stage, progress
            await session.commit()
            payload = task_view(task)
        await self._redis.set_task_state(state["task_id"], payload)

    async def _emit(self, state: FormatReviewState, stage: str, message: str, *, terminal: bool = False) -> int:
        async with self._sessions() as session:
            review_id = state.get("review_id")
            if not review_id:
                task = await session.get(TaskRecord, state["task_id"])
                review_id = task.resource_id if task is not None else None
            review = await session.scalar(
                select(FormatReview)
                .where(FormatReview.format_review_id == review_id)
                .with_for_update()
            )
            if review is None:
                raise FormatReviewFailure("FORMAT_REVIEW_NOT_FOUND", "格式审查任务不存在。")
            review.event_sequence += 1
            await session.commit()
            sequence = review.event_sequence
        await self._redis.append_event(
            StreamEvent(
                event_id=str(uuid4()),
                event_type="final" if terminal else "status",
                task_id=state["task_id"],
                message_id=str(review_id),
                session_id=str(review_id),
                sequence=sequence,
                timestamp=datetime.now(UTC),
                data={"stage": stage, "message": message},
            )
        )
        return sequence

    async def _check_cancelled(self, task_id: str) -> None:
        if await self._redis.is_cancelled(task_id):
            raise FormatReviewFailure("TASK_CANCELLED", "格式审查已取消。")

    async def _trace(
        self,
        task: dict[str, Any],
        node_name: str,
        started: float,
        status: str,
        *,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        async with self._sessions() as session:
            session.add(
                TraceRecord(
                    task_id=task["task_id"],
                    message_id=None,
                    request_id=str(task.get("request_id") or task["task_id"]),
                    correlation_id=str(task.get("correlation_id") or task["task_id"]),
                    node_name=node_name,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    status=status,
                    input_digest=_digest({"node": node_name, "task_id": task["task_id"]}),
                    output_digest=_digest(metrics or {}),
                    metrics_json=metrics or {},
                )
            )
            await session.commit()


def _legacy_plan_review_units(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create stable page/role units; fallback grouping is page structure, never token count."""

    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        page = int(fact.get("page_number") or 0)
        if page > 0:
            by_page[page].append(fact)
    pages = sorted(by_page)
    if not pages:
        return []

    units: list[dict[str, Any]] = []

    def append_unit(kind: str, title: str, selected_facts: list[dict[str, Any]]) -> None:
        if not selected_facts:
            return
        page_range = sorted({int(item["page_number"]) for item in selected_facts})
        units.append(
            {
                "unit_id": f"u-{len(units) + 1:03d}",
                "unit_position": len(units),
                "unit_kind": kind,
                "title": title,
                "page_range": [page_range[0], page_range[-1]],
                "block_ids": [str(item.get("block_id")) for item in selected_facts if item.get("block_id")],
                "fact_ids": [str(item["evidence_id"]) for item in selected_facts],
                "expected_rule_ids": [],
                "allocated_rule_ids": [],
                "global_rule_ids": [],
                "not_applicable_rule_ids": [],
                "retrieved_rule_ids": [],
                "coverage": {},
            }
        )

    first_page = by_page[pages[0]]
    append_unit("front_matter", "标题与投稿信息", first_page)
    abstract_facts = [fact for fact in facts if _fact_matches(fact, ("abstract", "摘要"))]
    append_unit("abstract", "摘要", abstract_facts or first_page)

    reference_pages = [
        page
        for page in pages
        if any(_fact_matches(fact, ("references", "bibliography", "参考文献")) for fact in by_page[page])
    ]
    reference_start = min(reference_pages) if reference_pages else None
    appendix_pages = _appendix_pages(facts)
    appendix_start = min(appendix_pages) if appendix_pages else None
    body_facts = [
        fact
        for fact in facts
        if int(fact.get("page_number") or 0) > pages[0]
        and (reference_start is None or int(fact.get("page_number") or 0) < reference_start)
    ]
    sections: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in body_facts:
        section = str(fact.get("section_title") or "").strip()
        if section and section.lower() not in {"abstract", "摘要"}:
            sections[section].append(fact)
    if sections:
        for title, section_facts in sorted(
            sections.items(), key=lambda item: min(int(fact["page_number"]) for fact in item[1])
        ):
            append_unit("body_section", title, section_facts)
    elif body_facts:
        append_unit("body_section", "正文", body_facts)

    figure_table_facts = [
        fact
        for fact in facts
        if _fact_matches(fact, ("figure", "fig.", "table", "图", "表"))
        or _is_native_graphic_fact(fact)
    ]
    if figure_table_facts:
        append_unit("figure_table", "图表与公式", figure_table_facts)
    if reference_start is not None:
        append_unit(
            "reference",
            "参考文献",
            [
                fact
                for fact in facts
                if int(fact.get("page_number") or 0) >= reference_start
                and (appendix_start is None or int(fact.get("page_number") or 0) < appendix_start)
            ],
        )
    appendix_facts = [fact for fact in facts if int(fact.get("page_number") or 0) in appendix_pages]
    if appendix_facts:
        append_unit("appendix", "附录", appendix_facts)

    # The only all-paper context unit. Its helper selects compact facts at
    # execution time; it does not make the entire PDF an LLM prompt.
    append_unit("global", "全篇与跨章节版面", facts)
    return units


def _plan_review_units(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Plan a small set of stable semantic units before any rule allocation.

    Extracted subheadings are evidence, not separate LLM jobs.  Adjacent
    first-level sections are combined into at most three body units.  When
    first-level headings are unavailable, consecutive pages provide the same
    non-token-count fallback boundary.
    """

    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        page = int(fact.get("page_number") or 0)
        if page > 0:
            by_page[page].append(fact)
    pages = sorted(by_page)
    if not pages:
        return []

    units: list[dict[str, Any]] = []

    def append_unit(kind: str, title: str, selected_facts: list[dict[str, Any]]) -> None:
        if not selected_facts:
            return
        page_range = sorted({int(item["page_number"]) for item in selected_facts})
        units.append(
            {
                "unit_id": f"u-{len(units) + 1:03d}",
                "unit_position": len(units),
                "unit_kind": kind,
                "title": title,
                "page_range": [page_range[0], page_range[-1]],
                "block_ids": [str(item.get("block_id")) for item in selected_facts if item.get("block_id")],
                "fact_ids": [str(item["evidence_id"]) for item in selected_facts],
                "expected_rule_ids": [],
                "allocated_rule_ids": [],
                "global_rule_ids": [],
                "not_applicable_rule_ids": [],
                "retrieved_rule_ids": [],
                "coverage": {},
            }
        )

    first_page = by_page[pages[0]]
    abstract_facts = [fact for fact in facts if _fact_matches(fact, ("abstract",))]
    abstract_ids = {str(fact.get("evidence_id")) for fact in abstract_facts}
    front_matter_facts = [
        fact
        for fact in first_page
        if str(fact.get("evidence_id")) not in abstract_ids
        and (
            str(fact.get("role") or "").lower()
            in {"title", "document_title", "author", "authors", "affiliation", "front_matter"}
            or not re.match(r"^\d+(?:\s|\.|$)", str(fact.get("section_title") or "").strip())
        )
    ]
    append_unit("front_matter", "Front matter", front_matter_facts or first_page)
    append_unit("abstract", "Abstract", abstract_facts or first_page)

    reference_pages = [
        page
        for page in pages
        if any(_fact_matches(fact, ("references", "bibliography")) for fact in by_page[page])
    ]
    reference_start = min(reference_pages) if reference_pages else None
    appendix_pages = _appendix_pages(facts)
    appendix_start = min(appendix_pages) if appendix_pages else None
    body_facts = [
        fact
        for fact in facts
        if str(fact.get("evidence_id")) not in abstract_ids
        and fact not in front_matter_facts
        and not str(fact.get("role") or "").startswith("derived_")
        if reference_start is None or int(fact.get("page_number") or 0) < reference_start
    ]
    for title, group_facts in _coarse_body_groups(body_facts):
        append_unit("body_section", title, group_facts)

    figure_table_facts = [
        fact
        for fact in facts
        if _fact_matches(fact, ("figure", "fig.", "table"))
        or _is_native_graphic_fact(fact)
    ]
    if figure_table_facts:
        append_unit("figure_table", "Figures and tables", figure_table_facts)
    if reference_start is not None:
        append_unit(
            "reference",
            "References",
            [
                fact
                for fact in facts
                if int(fact.get("page_number") or 0) >= reference_start
                and (appendix_start is None or int(fact.get("page_number") or 0) < appendix_start)
            ],
        )
    appendix_facts = [fact for fact in facts if int(fact.get("page_number") or 0) in appendix_pages]
    if appendix_facts:
        append_unit("appendix", "Appendix", appendix_facts)
    append_unit("global", "Global layout", facts)
    return units


def _appendix_pages(facts: list[dict[str, Any]]) -> set[int]:
    """Recognize appendix headers, not prose that merely mentions an appendix."""

    pages: set[int] = set()
    for fact in facts:
        page = int(fact.get("page_number") or 0)
        if page <= 0:
            continue
        title = str(fact.get("section_title") or "").strip()
        quote = str(fact.get("quote") or "").strip()
        role = str(fact.get("role") or "").lower()
        title_is_appendix = bool(
            title
            and re.match(r"^(?:appendix\b|[A-Z]\.\s+)", title, flags=re.IGNORECASE)
        )
        explicit_heading_is_appendix = bool(
            role in {"heading", "title", "section"}
            and re.match(r"^(?:appendix\b|[A-Z]\.\s+)", quote, flags=re.IGNORECASE)
        )
        # Native spans include reference authors such as "A. Smith". They
        # are useful layout evidence but cannot establish a document boundary.
        if title_is_appendix or explicit_heading_is_appendix:
            pages.add(page)
    if not pages:
        return set()
    last_page = max(int(item.get("page_number") or 0) for item in facts)
    return set(range(min(pages), last_page + 1))


def _coarse_body_groups(facts: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Keep top-level sections independent; otherwise use short page windows."""

    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    root_titles: dict[str, str] = {}
    for fact in facts:
        section_title = str(fact.get("section_title") or "").strip()
        match = re.match(r"^(\d+)(?:\s|\.|$)", section_title)
        if match is None:
            continue
        root = match.group(1)
        by_root[root].append(fact)
        # `4.1 Datasets and Features` belongs to chapter 4 but is not the
        # chapter title. Only a bare `4 Experiments` / `4. Experiments` span
        # can name the coarse body unit.
        if root not in root_titles and re.match(rf"^{re.escape(root)}(?:\s+|\.\s+)", section_title):
            root_titles[root] = section_title

    ordered_roots = sorted(
        by_root,
        key=lambda root: min(int(fact.get("page_number") or 0) for fact in by_root[root]),
    )
    if ordered_roots:
        root_start_pages = {
            root: min(int(fact.get("page_number") or 0) for fact in by_root[root])
            for root in ordered_roots
        }
        assigned_ids = {id(fact) for root in ordered_roots for fact in by_root[root]}
        for fact in facts:
            if id(fact) in assigned_ids:
                continue
            page = int(fact.get("page_number") or 0)
            eligible = [root for root in ordered_roots if root_start_pages[root] <= page]
            target = eligible[-1] if eligible else ordered_roots[0]
            by_root[target].append(fact)
        return [
            (root_titles.get(root, f"Section {root}"), by_root[root])
            for root in ordered_roots
        ]

    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        page = int(fact.get("page_number") or 0)
        if page > 0:
            by_page[page].append(fact)
    pages = sorted(by_page)
    if not pages:
        return []
    return [
        (
            f"Body pages {group_pages[0]}-{group_pages[-1]}",
            [fact for page in group_pages for fact in by_page[page]],
        )
        for start in range(0, len(pages), MAX_BODY_PAGES_PER_UNIT)
        for group_pages in [pages[start : start + MAX_BODY_PAGES_PER_UNIT]]
        if group_pages
    ]


def _allocate_rules_to_units(
    *,
    units: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    standard_evidences: list[dict[str, Any]],
    submission_mode: str,
    retrieval_coverage: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Produce auditable per-unit expected rule IDs and no-silent-omission coverage."""

    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        by_kind[str(unit["unit_kind"])].append(unit)
    global_units = by_kind.get("global", [])
    if not global_units:
        raise FormatReviewFailure("FORMAT_REVIEW_UNIT_PLAN_INVALID", "审查块计划缺少跨章节块。")
    global_unit = global_units[0]
    retrieved_rule_ids = {str(item.get("canonical_rule_id")) for item in standard_evidences}
    unallocated: list[str] = []
    for rule in manifest:
        rule_id = str(rule["rule_id"])
        applicable, condition_evidence_ids, reason = _rule_applicable_to_document(
            rule, facts, submission_mode
        )
        if not applicable:
            global_unit["not_applicable_rule_ids"].append(
                {
                    "rule_id": rule_id,
                    "condition_evidence_ids": condition_evidence_ids,
                    "reason": reason,
                }
            )
            continue
        kinds = set(rule.get("applicable_unit_kinds", []))
        is_global = bool(rule.get("is_global"))
        requires_cross_unit = bool(rule.get("requires_cross_unit"))
        target_units: list[dict[str, Any]]
        if is_global or requires_cross_unit:
            target_units = [global_unit]
        else:
            target_units = [unit for kind in kinds for unit in by_kind.get(str(kind), [])]
            if not target_units:
                # An active rule with no matching semantic unit must remain
                # observable as unverifiable rather than disappear silently.
                target_units = [global_unit]
        for target in target_units:
            target["expected_rule_ids"].append(rule_id)
            target["allocated_rule_ids"].append(rule_id)
            if is_global:
                target["global_rule_ids"].append(rule_id)
        if not target_units:
            unallocated.append(rule_id)

    for unit in units:
        expected = sorted(set(unit["expected_rule_ids"]))
        allocated = sorted(set(unit["allocated_rule_ids"]))
        retrieved = sorted(set(expected) & retrieved_rule_ids)
        missing = sorted(set(expected) - set(retrieved))
        unit["expected_rule_ids"] = expected
        unit["allocated_rule_ids"] = allocated
        unit["global_rule_ids"] = sorted(set(unit["global_rule_ids"]))
        unit["retrieved_rule_ids"] = retrieved
        unit["coverage"] = {
            "expected_rule_ids": expected,
            "retrieved_rule_ids": retrieved,
            "missing_rule_ids": missing,
            "complete": not missing,
        }

    required_rule_ids = {str(item["rule_id"]) for item in manifest}
    not_applicable_ids = {
        str(item["rule_id"])
        for unit in units
        for item in unit.get("not_applicable_rule_ids", [])
    }
    allocated_ids = {rule_id for unit in units for rule_id in unit["allocated_rule_ids"]}
    coverage = {
        **retrieval_coverage,
        "expected_rule_ids": sorted(required_rule_ids),
        "retrieved_rule_ids": sorted(retrieved_rule_ids),
        "missing_rule_ids": sorted(required_rule_ids - retrieved_rule_ids),
        "not_applicable_rule_ids": sorted(not_applicable_ids),
        "unallocated_rule_ids": sorted((required_rule_ids - not_applicable_ids) - allocated_ids) + unallocated,
    }
    return units, coverage


def _refine_units_for_context_budget(
    *,
    units: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    standards: list[dict[str, Any]],
    rules_by_id: dict[str, dict[str, Any]],
    submission_mode: str,
    context_budget: int = FORMAT_MODEL_INPUT_TOKENS,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Split a review unit until its complete rendered model input is at most 11K tokens.

    Atomic-rule boundaries are preferred so a rule is never duplicated across
    model calls. A single-rule leaf may then split on consecutive page
    boundaries. No rule text or cited fact is silently truncated.
    """

    refined_count = 0
    max_depth_reached = 0

    def refine(unit: dict[str, Any], depth: int) -> list[dict[str, Any]]:
        nonlocal refined_count, max_depth_reached
        max_depth_reached = max(max_depth_reached, depth)
        expected_rule_ids = sorted(
            {str(item) for item in unit.get("expected_rule_ids", [])}
        )
        if not expected_rule_ids:
            unit["context_refinement"] = {"depth": depth, "status": "not_needed"}
            return [unit]
        unit_facts = _facts_for_unit(unit, facts)
        unit_standards = [
            item
            for item in standards
            if str(item.get("canonical_rule_id")) in expected_rule_ids
        ]
        payload = _unit_context_payload(
            unit=unit,
            facts=unit_facts,
            standards=unit_standards,
            rules_by_id=rules_by_id,
            submission_mode=submission_mode,
        )
        context_tokens = format_review_input_tokens("format_check", payload)
        if context_tokens <= context_budget:
            unit["context_refinement"] = {
                "depth": depth,
                "status": "within_budget",
                "context_tokens": context_tokens,
                "token_limit": context_budget,
            }
            return [unit]
        children = _split_unit_by_rules(unit) if len(expected_rule_ids) > 1 else []
        if not children:
            children = _split_unit_at_page_boundary(unit, unit_facts)
        if depth >= MAX_CONTEXT_REFINEMENT_DEPTH or not children:
            unit["context_refinement"] = {
                "depth": depth,
                "status": "minimum_unit_exceeds_budget",
                "context_tokens": context_tokens,
                "token_limit": context_budget,
            }
            return [unit]
        refined_count += 1
        return [child for child in children for child in refine(child, depth + 1)]

    refined_units = [child for unit in units for child in refine(dict(unit), 0)]
    for position, unit in enumerate(refined_units):
        unit["unit_id"] = f"u-{position + 1:03d}"
        unit["unit_position"] = position
    return refined_units, {
        "split_parent_count": refined_count,
        "max_depth": max_depth_reached,
        "final_unit_count": len(refined_units),
    }


def _split_unit_by_rules(unit: dict[str, Any]) -> list[dict[str, Any]]:
    """Bisect atomic rules while preserving one shared inspection surface."""

    rule_ids = sorted({str(item) for item in unit.get("expected_rule_ids", [])})
    if len(rule_ids) < 2:
        return []
    split_at = len(rule_ids) // 2
    children: list[dict[str, Any]] = []
    for index, selected in enumerate((rule_ids[:split_at], rule_ids[split_at:]), start=1):
        selected_set = set(selected)
        coverage = dict(unit.get("coverage", {}))
        retrieved = sorted(selected_set & set(unit.get("retrieved_rule_ids", [])))
        coverage.update(
            {
                "expected_rule_ids": selected,
                "retrieved_rule_ids": retrieved,
                "missing_rule_ids": sorted(selected_set - set(retrieved)),
                "complete": selected_set <= set(retrieved),
            }
        )
        children.append(
            {
                **unit,
                "title": f"{unit['title']} (rules {index}/2)",
                "expected_rule_ids": selected,
                "allocated_rule_ids": selected,
                "global_rule_ids": sorted(selected_set & set(unit.get("global_rule_ids", []))),
                "retrieved_rule_ids": retrieved,
                # Document-level non-applicability is emitted once, not once
                # per rule child.
                "not_applicable_rule_ids": (
                    list(unit.get("not_applicable_rule_ids", [])) if index == 1 else []
                ),
                "coverage": coverage,
                "context_refinement": {"status": "rule_split_pending"},
            }
        )
    return children


def _split_unit_at_page_boundary(
    unit: dict[str, Any], unit_facts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Create two consecutive page children while preserving full rule scope."""

    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for fact in unit_facts:
        page = int(fact.get("page_number") or 0)
        if page > 0:
            by_page[page].append(fact)
    pages = sorted(by_page)
    if len(pages) < 2:
        return []
    split_at = len(pages) // 2
    page_groups = [pages[:split_at], pages[split_at:]]
    children: list[dict[str, Any]] = []
    for index, group_pages in enumerate(page_groups, start=1):
        selected = [fact for page in group_pages for fact in by_page[page]]
        if not selected:
            continue
        children.append(
            {
                **unit,
                "title": f"{unit['title']} (refined {index}/2)",
                "page_range": [group_pages[0], group_pages[-1]],
                "fact_ids": [str(fact["evidence_id"]) for fact in selected],
                "block_ids": [
                    str(fact.get("block_id")) for fact in selected if fact.get("block_id")
                ],
                "context_refinement": {"status": "split_pending"},
            }
        )
    return children


def _rule_applicable_to_document(
    rule: dict[str, Any], facts: list[dict[str, Any]], submission_mode: str
) -> tuple[bool, list[str], str | None]:
    conditions = rule.get("applicability_conditions")
    if not isinstance(conditions, dict):
        return True, [], None
    mode_conditions = {str(value) for value in conditions.get("requires_submission_mode", [])}
    if mode_conditions and submission_mode not in mode_conditions:
        return False, [], "投稿模式不满足规则适用条件。"
    object_types = {str(value).lower() for value in conditions.get("requires_object_types", [])}
    if object_types:
        matching = [
            fact
            for fact in facts
            if any(_fact_matches(fact, (object_type,)) for object_type in object_types)
            or (
                "figure" in object_types
                and str(fact.get("role") or "")
                in {"native_image_object", "native_vector_graphic_object"}
            )
        ]
        if not matching:
            return False, [], "论文不包含该规则要求的对象类型。"
        return True, [str(item["evidence_id"]) for item in matching[:8]], None
    section_roles = {str(value).lower() for value in conditions.get("requires_section_roles", [])}
    if section_roles:
        matching = [
            fact
            for fact in facts
            if str(fact.get("role") or "").lower() in section_roles
            or str(fact.get("section_title") or "").lower() in section_roles
        ]
        if not matching:
            return False, [], "论文不包含该规则要求的章节角色。"
        return True, [str(item["evidence_id"]) for item in matching[:8]], None
    return True, [], None


def _facts_for_unit(unit: dict[str, Any], facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fact_ids = {str(item) for item in unit.get("fact_ids", [])}
    selected = [fact for fact in facts if str(fact.get("evidence_id")) in fact_ids]
    supporting_roles_by_unit = {
        "figure_table": {
            "derived_captions",
            "derived_caption_geometry",
            "derived_column_geometry",
            "derived_body_geometry",
            "derived_rule_assessment",
        },
        "front_matter": {
            "derived_front_matter",
            "derived_front_matter_geometry",
            "derived_supplement_front_matter",
            "derived_column_geometry",
            "derived_rule_assessment",
        },
        "body_section": {
            "derived_column_geometry",
            "derived_typography_inventory",
            "derived_heading_inventory",
            "derived_supplement_heading_case",
            "derived_citation_inventory",
            "derived_rule_assessment",
        },
        "abstract": {
            "derived_abstract",
            "derived_rule_assessment",
        },
        "appendix": {
            "derived_page_geometry",
            "derived_document_structure",
            "derived_column_geometry",
            "derived_typography_inventory",
            "derived_appendix_layout",
        },
        "reference": {
            "derived_references",
            "derived_reference_inventory",
            "derived_supplement_reference_indentation",
            "derived_rule_assessment",
        },
        "global": {
            "derived_page_geometry",
            "derived_document_structure",
            "derived_column_geometry",
            "derived_typography_inventory",
            "derived_appendix_layout",
            "derived_supplement_page_boundary",
            "derived_rule_assessment",
        },
    }
    supporting_roles = supporting_roles_by_unit.get(str(unit.get("unit_kind") or ""), set())
    if supporting_roles:
        # Cross-block aggregates are document-derived and are not owned by one
        # local text block. Inject their compact source facts for every unit
        # whose atomic rules require their complete coverage.
        selected_ids = {str(item.get("evidence_id")) for item in selected}
        selected.extend(
            fact
            for fact in facts
            if str(fact.get("role")) in supporting_roles
            and str(fact.get("evidence_id")) not in selected_ids
        )
    if unit.get("unit_kind") == "global":
        selected = _global_context_facts(selected)
    return selected


def _global_context_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Venue-derived measurements are the authoritative evidence for global
    # geometry rules. Keep them before page sampling so ordinary text blocks
    # cannot consume the global context budget first.
    selected: list[dict[str, Any]] = [
        fact for fact in facts if str(fact.get("role") or "").startswith("derived_")
    ]
    selected = selected[:MAX_CONTEXT_FACTS]
    selected_ids = {str(item.get("evidence_id")) for item in selected}
    seen_pages: set[int] = set()
    for fact in sorted(facts, key=lambda item: (int(item.get("page_number") or 0), str(item.get("evidence_id")))):
        if str(fact.get("evidence_id")) in selected_ids:
            continue
        page = int(fact.get("page_number") or 0)
        if page and page not in seen_pages and fact.get("page_width_pt") is not None:
            selected.append(fact)
            selected_ids.add(str(fact.get("evidence_id")))
            seen_pages.add(page)
        if len(selected) >= MAX_CONTEXT_FACTS:
            return selected
    for fact in facts:
        if str(fact.get("evidence_id")) in selected_ids:
            continue
        if _fact_matches(fact, ("author", "acknowledg", "references", "bibliography", "page")):
            selected.append(fact)
            selected_ids.add(str(fact.get("evidence_id")))
        if len(selected) >= MAX_CONTEXT_FACTS:
            break
    return selected[:MAX_CONTEXT_FACTS]


def _unit_context_payload(
    *,
    unit: dict[str, Any],
    facts: list[dict[str, Any]],
    standards: list[dict[str, Any]],
    rules_by_id: dict[str, dict[str, Any]],
    submission_mode: str,
) -> dict[str, Any]:
    categories = sorted(
        {str(rules_by_id[rule_id]["rule_category"]) for rule_id in unit["expected_rule_ids"] if rule_id in rules_by_id}
    )
    context_facts = _facts_for_expected_neurips_rule_groups(
        facts, unit["expected_rule_ids"], rules_by_id
    )
    selected_facts = (
        _context_facts_for_categories(
            context_facts, categories, per_category_limit=FORMAT_CONTEXT_FACTS_PER_CATEGORY
        )
        if categories
        else []
    )
    if not selected_facts:
        selected_facts = [_context_fact(item) for item in context_facts[:MAX_CONTEXT_FACTS]]
    return {
        "format_profile": {"submission_mode": submission_mode},
        "review_unit": _model_unit_plan(unit),
        "target_categories": categories,
        "expected_rule_ids": unit["expected_rule_ids"],
        "review_policy_by_rule": {
            rule_id: _rule_review_policy(rules_by_id[rule_id])
            for rule_id in unit["expected_rule_ids"]
            if rule_id in rules_by_id
        },
        "standard_evidence": [_model_standard_evidence(item) for item in standards],
        "paper_layout_facts": selected_facts,
        "required_output": "Composite findings for this unit using only supplied P*/S* evidence identifiers.",
    }


def _facts_for_expected_neurips_rule_groups(
    facts: list[dict[str, Any]], expected_rule_ids: list[str], rules_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Avoid repeating unrelated NeurIPS rule-group measurements in each unit.

    The mapping is derived from the manifest's stable NIPS-xx section marker,
    not from a document title, paper content, or a particular rule hash.
    """

    expected_groups: set[str] = set()
    for rule_id in expected_rule_ids:
        rule = rules_by_id.get(rule_id)
        if not isinstance(rule, dict):
            continue
        source = " ".join(
            str(rule.get(key) or "") for key in ("section_path", "title", "description")
        )
        expected_groups.update(re.findall(r"\bNIPS-\d{2}\b", source, flags=re.IGNORECASE))
    if not expected_groups:
        return facts
    normalized_groups = {item.upper() for item in expected_groups}
    return [
        fact
        for fact in facts
        if str(fact.get("role")) != "derived_rule_assessment"
        or str(
            (fact.get("measurements") if isinstance(fact.get("measurements"), dict) else {}).get(
                "parser_rule_group"
            )
            or ""
        ).upper()
        in normalized_groups
    ]


def _not_applicable_findings(
    unit: dict[str, Any],
    rules_by_id: dict[str, dict[str, Any]],
    standards: list[dict[str, Any]],
    all_facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    standards_by_rule = {str(item.get("canonical_rule_id")): item for item in standards}
    facts_by_id = {str(item.get("evidence_id")): item for item in all_facts}
    findings: list[dict[str, Any]] = []
    for item in unit.get("not_applicable_rule_ids", []):
        rule_id = str(item.get("rule_id"))
        rule = rules_by_id.get(rule_id)
        if rule is None:
            continue
        condition_facts = [
            facts_by_id[fact_id]
            for fact_id in item.get("condition_evidence_ids", [])
            if fact_id in facts_by_id
        ]
        standard = standards_by_rule.get(rule_id)
        findings.append(
            {
                "rule_ids": [rule_id],
                "category": rule["rule_category"],
                "aspect": rule["title"],
                "result": "not_applicable",
                "severity": "info",
                "finding": str(item.get("reason") or "该规则的结构化适用条件不满足。"),
                "suggestion": None,
                "paper_evidences": condition_facts,
                "standard_evidences": [standard] if standard else [],
                "evidence_status": "complete" if standard else "incomplete",
                "reason": str(item.get("reason") or ""),
                "unit_id": unit["unit_id"],
                "unit_position": unit["unit_position"],
            }
        )
    return findings


def _unverifiable_findings_for_rules(
    rule_ids: list[str],
    rules_by_id: dict[str, dict[str, Any]],
    reason: str,
    *,
    facts: list[dict[str, Any]] | None = None,
    standards: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for rule_id in rule_ids:
        rule = rules_by_id.get(rule_id)
        if rule is None:
            continue
        category = str(rule["rule_category"])
        # These facts are retained as the inspected context for an incomplete
        # result, never promoted to proof of compliance or non-compliance.
        paper_evidences = _context_facts_for_category(facts or [], category, limit=3)
        standard_evidences = [
            item
            for item in standards or []
            if str(item.get("canonical_rule_id")) == rule_id
        ]
        findings.append(
            {
                "rule_ids": [rule_id],
                "category": category,
                "aspect": rule["title"],
                "result": "unverifiable",
                "severity": "info",
                "finding": reason,
                "suggestion": None,
                "paper_evidences": paper_evidences,
                "standard_evidences": standard_evidences,
                "evidence_status": "incomplete",
                "reason": reason,
            }
        )
    return findings


def _ensure_unit_rule_representation(
    findings: list[dict[str, Any]],
    rule_ids: list[str],
    rules_by_id: dict[str, dict[str, Any]],
    *,
    facts: list[dict[str, Any]] | None = None,
    standards: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    represented_rule_ids = {
        str(rule_id)
        for item in findings
        for rule_id in item.get("rule_ids", [])
        if str(rule_id)
    }
    represented_rule_ids.update(
        str(evidence.get("canonical_rule_id"))
        for item in findings
        for evidence in item.get("standard_evidences", [])
        if evidence.get("canonical_rule_id")
    )
    missing = [
        rule_id
        for rule_id in rule_ids
        if rule_id in rules_by_id and rule_id not in represented_rule_ids
    ]
    return [
        *findings,
        *_unverifiable_findings_for_rules(
            missing,
            rules_by_id,
            "模型未形成可验证的块级发现。",
            facts=facts,
            standards=standards,
        ),
    ]


def _deduplicate_unit_findings(unit_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    without_rule: list[dict[str, Any]] = []
    for result in sorted(unit_results, key=lambda item: int(item.get("unit_position") or 0)):
        for finding in result.get("findings", []):
            enriched = {
                **finding,
                "unit_id": finding.get("unit_id") or result["unit_id"],
                "unit_position": finding.get("unit_position", result["unit_position"]),
            }
            rule_ids = {
                str(rule_id)
                for rule_id in enriched.get("rule_ids", [])
                if str(rule_id)
            }
            rule_ids.update(
                str(evidence.get("canonical_rule_id"))
                for evidence in enriched.get("standard_evidences", [])
                if evidence.get("canonical_rule_id")
            )
            if not rule_ids:
                without_rule.append(enriched)
                continue
            for rule_id in sorted(rule_ids):
                by_rule[rule_id].append({**enriched, "rule_ids": [rule_id]})

    findings = [
        _consolidate_rule_findings(rule_id, candidates)
        for rule_id, candidates in sorted(by_rule.items())
    ]
    seen_without_rule: set[tuple[str, str, str]] = set()
    for finding in without_rule:
        key = (
            str(finding.get("category")),
            str(finding.get("aspect")),
            str(finding.get("result")),
        )
        if key not in seen_without_rule:
            findings.append(finding)
            seen_without_rule.add(key)
    return findings


def _reflection_payload(
    unit: dict[str, Any], findings: list[dict[str, Any]], coverage: dict[str, Any]
) -> dict[str, Any]:
    """Keep evidence reflection within the same request budget as checking."""

    def evidence_ids(item: dict[str, Any], field: str) -> list[str]:
        return [
            str(evidence.get("evidence_id"))
            for evidence in item.get(field, [])
            if isinstance(evidence, dict) and evidence.get("evidence_id")
        ][:8]

    return {
        "unit": _public_unit_plan(unit),
        "findings": [
            {
                "rule_ids": item.get("rule_ids", []),
                "result": item.get("result"),
                "severity": item.get("severity"),
                "evidence_status": item.get("evidence_status"),
                "paper_evidence_ids": evidence_ids(item, "paper_evidences"),
                "standard_evidence_ids": evidence_ids(item, "standard_evidences"),
                "finding": str(item.get("finding") or "")[:120],
                "reason": str(item.get("reason") or "")[:80],
            }
            for item in findings
        ],
        "coverage_report": _compact_reflection_coverage(coverage),
    }


def _compact_reflection_coverage(coverage: dict[str, Any]) -> dict[str, Any]:
    return {
        key: list(coverage.get(key, []))[:32]
        for key in (
            "expected_rule_ids",
            "retrieved_rule_ids",
            "missing_rule_ids",
            "missing_categories",
        )
        if isinstance(coverage.get(key), list)
    }


def _synthesis_payload(findings: list[dict[str, Any]], coverage: dict[str, Any]) -> dict[str, Any]:
    """Use a compact deterministic inventory for the optional LLM summary."""

    return {
        "validated_unit_findings": [
            {
                "rule_ids": item.get("rule_ids", []),
                "category": item.get("category"),
                "aspect": str(item.get("aspect") or item.get("rule_title") or "")[:120],
                "result": item.get("result"),
                "severity": item.get("severity"),
                "page_numbers": sorted(
                    {
                        int(evidence.get("page_number") or evidence.get("page"))
                        for evidence in item.get("paper_evidences", [])
                        if isinstance(evidence, dict)
                        and isinstance(
                            evidence.get("page_number", evidence.get("page")), int
                        )
                    }
                ),
            }
            for item in findings
        ],
        "coverage_report": _compact_reflection_coverage(coverage),
        "constraint": "Only summarize supplied findings. Do not add findings or evidence.",
    }


def _consolidate_rule_findings(
    rule_id: str, candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    result_priority = {
        "not_applicable": 0,
        "unverifiable": 1,
        "compliant": 2,
        "non_compliant": 3,
    }
    chosen_result = max(
        (str(item.get("result")) for item in candidates),
        key=lambda value: result_priority.get(value, -1),
    )
    selected = [item for item in candidates if str(item.get("result")) == chosen_result]
    base = selected[0]
    severity_priority = {"info": 0, "low": 1, "medium": 2, "high": 3}
    severity = max(
        (str(item.get("severity") or "info") for item in selected),
        key=lambda value: severity_priority.get(value, -1),
    )

    def merge_evidence(field: str) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in selected:
            for evidence in item.get(field, []):
                key = str(evidence.get("evidence_id") or json.dumps(evidence, sort_keys=True))
                if key not in seen:
                    merged.append(evidence)
                    seen.add(key)
        return merged

    # A canonical rule has one user-facing outcome. Multiple units or a
    # bounded retry can contribute extra evidence, but must not concatenate
    # near-identical prose into a paragraph-sized "finding".
    finding = str(base.get("finding") or "")
    reason = next((str(item["reason"]) for item in selected if item.get("reason")), None)
    suggestion = next(
        (str(item["suggestion"]) for item in selected if item.get("suggestion")), None
    )
    return {
        **base,
        "rule_ids": [rule_id],
        "result": chosen_result,
        "severity": severity,
        "finding": finding[:4000],
        "suggestion": suggestion[:4000] if suggestion else None,
        "reason": reason[:2000] if reason else None,
        "paper_evidences": merge_evidence("paper_evidences"),
        "standard_evidences": merge_evidence("standard_evidences"),
        "evidence_status": (
            "complete"
            if selected and all(item.get("evidence_status") == "complete" for item in selected)
            else "incomplete"
        ),
        "unit_id": None,
        "unit_position": min(int(item.get("unit_position") or 0) for item in selected),
    }


def _public_unit_plan(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        key: unit.get(key)
        for key in (
            "unit_id",
            "unit_position",
            "unit_kind",
            "title",
            "page_range",
            "block_ids",
            "expected_rule_ids",
            "allocated_rule_ids",
            "global_rule_ids",
            "not_applicable_rule_ids",
            "retrieved_rule_ids",
            "coverage",
            "status",
            "unit_cycle_count",
            "retry_budget_remaining",
            "last_retry_reason",
            "event_sequence",
            "context_refinement",
        )
        if key in unit
    }


def _fact_matches(fact: dict[str, Any], terms: tuple[str, ...]) -> bool:
    haystack = " ".join(
        [
            str(fact.get("quote") or "").lower(),
            str(fact.get("role") or "").lower(),
            str(fact.get("section_title") or "").lower(),
        ]
    )
    return any(term.lower() in haystack for term in terms)


def _is_native_graphic_fact(fact: dict[str, Any]) -> bool:
    return str(fact.get("role") or "") in {
        "native_image_object",
        "native_vector_graphic_object",
    }


def _applicable_manifest(raw: list[Any], submission_mode: str) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "active") != "active":
            continue
        mode = str(item.get("submission_mode") or "shared")
        if mode not in {"shared", submission_mode}:
            continue
        rule_id = str(item.get("canonical_rule_id") or item.get("rule_id") or "").strip()
        description = str(item.get("rule_text") or item.get("description") or "").strip()
        if not rule_id or not description:
            continue
        manifest.append(
            {
                "rule_id": rule_id,
                "title": str(item.get("title") or item.get("section_path") or rule_id),
                "description": description,
                "source_attachment": str(item.get("source_attachment") or "").strip(),
                "rule_category": str(item.get("rule_category") or "body"),
                "submission_mode": mode,
                "source_document_id": str(item.get("source_document_id") or ""),
                "section_path": str(item.get("section_path") or item.get("title") or ""),
                "keywords": item.get("keywords") if isinstance(item.get("keywords"), list) else [],
                "applicable_unit_kinds": list(item.get("applicable_unit_kinds") or []),
                "is_global": bool(item.get("is_global")),
                "requires_cross_unit": bool(item.get("requires_cross_unit")),
                "cross_unit_kinds": list(item.get("cross_unit_kinds") or []),
                "applicability_conditions": item.get("applicability_conditions")
                if isinstance(item.get("applicability_conditions"), dict)
                else {},
                "evidence_selector": list(item.get("evidence_selector") or []),
                "assessment_mode": str(item.get("assessment_mode") or "strict"),
                "supported_checks": list(item.get("supported_checks") or []),
            }
        )
    return manifest


def _category_query(snapshot: dict[str, Any], category: str, manifest: list[dict[str, Any]]) -> str:
    rules = [item for item in manifest if item["rule_category"] == category]
    terms = [category, *[item["title"] for item in rules], *[str(value) for item in rules for value in item["keywords"]]]
    return " ".join(
        [str(snapshot.get("venue_id") or snapshot.get("profile_key")), str(snapshot.get("format_version") or snapshot.get("version")), str(snapshot.get("submission_mode")), *terms]
    )


def _manifest_standard_evidences(
    manifest: list[dict[str, Any]], document_ids: list[str]
) -> list[dict[str, Any]]:
    """Materialize every selected atomic rule; semantic top-k must not gate coverage."""

    allowed_documents = set(document_ids)
    evidences: list[dict[str, Any]] = []
    for rule in manifest:
        document_id = str(rule.get("source_document_id") or "")
        rule_id = str(rule.get("rule_id") or "")
        quote = str(rule.get("description") or "").strip()
        if not rule_id or not quote or document_id not in allowed_documents:
            continue
        evidences.append(
            {
                "evidence_id": f"S{len(evidences) + 1}",
                "canonical_rule_id": rule_id,
                "category": str(rule.get("rule_category") or "body"),
                "document_id": document_id,
                "chunk_id": rule_id,
                "section_path": str(rule.get("section_path") or rule.get("title") or ""),
                "quote": quote,
                "source_attachment": str(rule.get("source_attachment") or ""),
                "source_uri": f"manifest://{document_id}/{rule_id}",
                "retrieval_score": 1.0,
                "retrieval_strategy": "manifest_exact",
            }
        )
    return evidences


def _resolve_standard_chunks(raw: list[dict[str, Any]], plan: dict[str, Any], existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_by_id = {str(item["rule_id"]): item for item in plan["manifest"]}
    allowed_documents = set(plan["document_ids"])
    known = {str(item.get("canonical_rule_id")) for item in existing}
    resolved: list[dict[str, Any]] = []
    for item in raw:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        document_id = str(item.get("document_id") or metadata.get("document_id") or "")
        supplied_rule_id = str(
            item.get("canonical_rule_id")
            or metadata.get("canonical_rule_id")
            or item.get("chunk_id")
            or item.get("id")
            or ""
        )
        content = str(item.get("content") or item.get("text") or "").strip()
        explicit_rule_id = _explicit_rule_id(content, set(manifest_by_id))
        canonical_id = (
            supplied_rule_id
            if supplied_rule_id in manifest_by_id
            else explicit_rule_id or _match_manifest_rule_id(content, document_id, plan["manifest"])
        )
        if document_id not in allowed_documents or not canonical_id or canonical_id in known:
            continue
        if not content:
            continue
        rule = manifest_by_id[canonical_id]
        evidence_id = f"S{len(existing) + len(resolved) + 1}"
        resolved.append(
            {
                "evidence_id": evidence_id,
                "canonical_rule_id": canonical_id,
                "category": rule["rule_category"],
                "document_id": document_id,
                "chunk_id": str(item.get("chunk_id") or item.get("id") or canonical_id),
                "section_path": metadata.get("section_path") or metadata.get("section") or rule["section_path"],
                # RAGFlow proves that this atomic chunk exists in the frozen
                # document. The profile manifest remains the canonical text
                # shown to the model, avoiding stale or oversized chunk prose.
                "quote": str(rule["description"]),
                "source_attachment": str(rule.get("source_attachment") or ""),
                "source_uri": metadata.get("source_uri"),
                "retrieval_score": item.get("score", item.get("similarity", 0.0)),
            }
        )
        known.add(canonical_id)
    return resolved


def _match_manifest_rule_id(
    content: str, document_id: str, manifest: list[dict[str, Any]]
) -> str | None:
    """Deterministically recover a canonical ID when RAGFlow omits chunk metadata."""

    content_key = _text_key(content)
    matches: list[str] = []
    for rule in manifest:
        description_key = _text_key(str(rule.get("description") or ""))
        if not description_key:
            continue
        # RAGFlow returns the imported rule wrapper in some versions and only
        # the cleaned rule prose in others. Both forms retain one text inside
        # the other, so this remains auditable rather than semantic matching.
        if description_key in content_key or content_key in description_key:
            matches.append(str(rule["rule_id"]))
    return matches[0] if len(matches) == 1 else None


def _explicit_rule_id(content: str, allowed_rule_ids: set[str]) -> str | None:
    """Read the canonical ID embedded in an imported atomic-rule chunk wrapper."""

    for line in content.splitlines()[:8]:
        match = re.match(r"^\s*(?:规则ID|rule\s*id)\s*[:：]\s*(\S+)\s*$", line, re.IGNORECASE)
        if match and match.group(1) in allowed_rule_ids:
            return match.group(1)
    return None


def _context_groups(
    plan: dict[str, Any], standards: list[dict[str, Any]], facts: list[dict[str, Any]]
) -> list[tuple[str, dict[str, Any] | None]]:
    payload = {
        "format_profile": {
            "venue_id": plan.get("venue_id"),
            "submission_mode": plan["submission_mode"],
        },
        "target_categories": plan["target_categories"],
        "standard_evidence": standards,
        "paper_layout_facts": facts,
        "required_output": "Composite findings with only supplied P*/S* evidence identifiers.",
    }
    if len(json.dumps(payload, ensure_ascii=False)) <= MAX_CONTEXT_CHARACTERS:
        return [("all", payload)]

    composite_facts = _context_facts_for_categories(facts, plan["target_categories"])
    composite_payload = {
        **payload,
        "paper_layout_facts": composite_facts,
        "fact_selection": {
            "strategy": "category_relevance_with_page_coverage",
            "available_fact_count": len(facts),
            "selected_fact_count": len(composite_facts),
        },
    }
    if len(json.dumps(composite_payload, ensure_ascii=False)) <= MAX_CONTEXT_CHARACTERS:
        return [("all", composite_payload)]

    groups: list[tuple[str, dict[str, Any] | None]] = []
    for category in plan["target_categories"]:
        category_payload = {
            **payload,
            "target_categories": [category],
            "standard_evidence": [item for item in standards if item["category"] == category],
            "paper_layout_facts": _context_facts_for_category(facts, category),
        }
        # Rules are never trimmed. If a complete rule category plus its selected
        # observable PDF facts does not fit, the caller records it as unverifiable.
        groups.append(
            (
                category,
                category_payload
                if len(json.dumps(category_payload, ensure_ascii=False)) <= MAX_CONTEXT_CHARACTERS
                else None,
            )
        )
    return groups


def _context_facts_for_categories(
    facts: list[dict[str, Any]],
    categories: list[str],
    *,
    per_category_limit: int | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for category in categories:
        for fact in _context_facts_for_category(
            facts, category, limit=per_category_limit or MAX_CONTEXT_FACTS
        ):
            evidence_id = str(fact.get("evidence_id") or "")
            if not evidence_id or evidence_id in seen:
                continue
            selected.append(fact)
            seen.add(evidence_id)
            if len(selected) >= MAX_COMPOSITE_CONTEXT_FACTS:
                return selected
    return selected


def _context_facts_for_category(
    facts: list[dict[str, Any]], category: str, *, limit: int = MAX_CONTEXT_FACTS
) -> list[dict[str, Any]]:
    """Pick deterministic, category-relevant PDF observations for an LLM context."""

    terms_by_category = {
        "anonymity": ("author", "affiliation", "email", "acknowledg", "funding", "@"),
        "author_identity": (
            "affiliation",
            "email",
            "@",
            "university",
            "school",
            "institute",
            "department",
            "corresponding",
        ),
        "abstract": ("abstract", "摘要"),
        "figure": ("figure", "fig.", "图", "caption"),
        "heading": ("heading", "section", "introduction", "conclusion", "references", "标题"),
        "reference": ("reference", "bibliography", "citation", "参考文献", "引用"),
        "table": ("table", "表格", "表 ", "tab."),
        "template": ("neurips", "latex", "style", "template", "submission"),
    }
    terms = terms_by_category.get(category, ())
    ranked: list[tuple[int, int, int, str, dict[str, Any]]] = []
    for fact in facts:
        role = str(fact.get("role") or "")
        if role.startswith("derived_") and not _derived_fact_relevant(fact, category):
            continue
        if (
            category in {"figure", "table"}
            and role == "paragraph"
            and not _caption_label(str(fact.get("quote") or ""))
        ):
            # Narrative mentions of a figure/table do not establish its label,
            # style, pairing, or geometry.  Keep the linked structured facts.
            continue
        quote = str(fact.get("quote") or "")
        haystack = " ".join(
            [
                quote.lower(),
                str(fact.get("role") or "").lower(),
                str(fact.get("section_title") or "").lower(),
            ]
        )
        term_score = sum(8 for term in terms if term in haystack)
        native_graphic = category in {"figure", "table"} and _is_native_graphic_fact(fact)
        if (
            category in {"figure", "table"}
            and not term_score
            and not native_graphic
            and not _derived_fact_relevant(fact, category)
        ):
            continue
        score = term_score
        if _derived_fact_relevant(fact, category):
            score += 40
        if native_graphic:
            score += 8
        if category == "page_layout" and fact.get("page_width_pt") is not None:
            score += 6
        if category == "heading" and fact.get("font_size_pt") is not None:
            score += 2
        if fact.get("source") == "native_pdf+mineru":
            score += 3
        if fact.get("role") != "native_text_span":
            score += 1
        ranked.append(
            (
                score,
                1 if fact.get("role") == "native_text_span" else 0,
                int(fact.get("page_number") or 0),
                str(fact.get("evidence_id") or ""),
                fact,
            )
        )

    ranked.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    # Structured measurements frequently share a page anchor. They are
    # independent facts and must not eliminate one another via page coverage.
    derived_ranked = [item for item in ranked if _derived_fact_relevant(item[4], category)]
    selected: list[dict[str, Any]] = [_context_fact(item[4]) for item in derived_ranked[:limit]]
    selected_ids: set[str] = {str(item.get("evidence_id") or "") for item in selected}
    # The NeurIPS extractor provides category-routable, complete rule-group
    # measurements.  Their compact derived inventories retain the geometry and
    # style observables, so unrelated body prose adds token cost without
    # increasing the evidence population.  Other venues keep the established
    # page-covering raw-fact fallback.
    if any(str(item.get("role")) == "derived_rule_assessment" for item in selected):
        return selected
    if len(selected) >= limit:
        return selected
    covered_pages: set[int] = set()
    for _, _, page_number, evidence_id, fact in ranked:
        if evidence_id in selected_ids:
            continue
        if page_number in covered_pages:
            continue
        context_fact = _context_fact(fact)
        selected.append(context_fact)
        selected_ids.add(evidence_id)
        covered_pages.add(page_number)
        if len(selected) >= limit:
            return selected

    for score, _, _, evidence_id, fact in ranked:
        if evidence_id in selected_ids or score <= 0:
            continue
        context_fact = _context_fact(fact)
        selected.append(context_fact)
        selected_ids.add(evidence_id)
        if len(selected) >= limit:
            break
    return selected


def _derived_fact_relevant(fact: dict[str, Any], category: str) -> bool:
    role = str(fact.get("role") or "")
    if not role.startswith("derived_"):
        return False
    name = role.removeprefix("derived_")
    if name == "rule_assessment":
        measurements = fact.get("measurements")
        categories = measurements.get("rule_categories") if isinstance(measurements, dict) else []
        return category in categories if isinstance(categories, list) else False
    if category == "page_layout":
        return name in {
            "page_geometry",
            "document_structure",
            "column_geometry",
            "body_geometry",
            "paragraph_metrics",
            "typography_inventory",
            "appendix_layout",
        }
    if category == "appendix":
        return name in {
            "page_geometry",
            "document_structure",
            "column_geometry",
            "typography_inventory",
            "appendix_layout",
        }
    if category in {"author_identity", "anonymity"}:
        return name in {"front_matter", "front_matter_geometry", "column_geometry"}
    if category == "heading":
        return name in {"front_matter", "front_matter_geometry", "heading_inventory", "column_geometry"}
    if category == "reference":
        return name in {"references", "citation_inventory", "reference_inventory"}
    if category in {"figure", "table"}:
        return name in {"captions", "caption_geometry", "column_geometry", "body_geometry"}
    return category in name or name in category


def _context_fact(fact: dict[str, Any]) -> dict[str, Any]:
    context_fact = dict(fact)
    if str(context_fact.get("role") or "").startswith("derived_"):
        measurements = context_fact.get("measurements")
        if isinstance(measurements, dict):
            context_fact["measurements"] = _compact_derived_measurements(
                str(context_fact.get("role") or ""), measurements
            )
        context_fact["quote"] = str(context_fact.get("role") or "derived measurement")
        for key in ("block_id", "section_title", "confidence", "source_uri"):
            context_fact.pop(key, None)
        return context_fact
    role = str(context_fact.get("role") or "")
    if role in {"figure_caption", "table_caption"}:
        return {
            key: context_fact.get(key)
            for key in (
                "evidence_id",
                "page_number",
                "bbox",
                "role",
                "font_name",
                "font_size_pt",
                "is_bold",
                "baseline_gap_pt",
                "page_width_pt",
                "page_height_pt",
            )
        } | {"caption_label": _caption_label(str(context_fact.get("quote") or ""))}
    if role in {
        "figure_object",
        "table_object",
        "native_image_object",
        "native_vector_drawing_object",
        "native_vector_graphic_object",
    }:
        return {
            key: context_fact.get(key)
            for key in (
                "evidence_id",
                "page_number",
                "bbox",
                "role",
                "page_width_pt",
                "page_height_pt",
            )
        }
    quote = str(context_fact.get("quote") or "")
    if len(quote) > MAX_CONTEXT_QUOTE_CHARACTERS:
        context_fact["quote"] = quote[:MAX_CONTEXT_QUOTE_CHARACTERS]
        context_fact["quote_truncated"] = True
    return context_fact


def _compact_derived_measurements(role: str, value: dict[str, Any]) -> dict[str, Any]:
    """Remove repeated raw arrays while retaining every global-rule measurement."""

    compact = dict(value)
    if role == "derived_page_geometry":
        pages = [item for item in compact.get("pages", []) if isinstance(item, dict)]
        unique_sizes = sorted(
            {
                (
                    float(item.get("width_pt") or 0),
                    float(item.get("height_pt") or 0),
                    int(item.get("rotation") or 0),
                )
                for item in pages
            }
        )
        compact["page_sizes"] = [
            {"width_pt": width, "height_pt": height, "rotation": rotation}
            for width, height, rotation in unique_sizes
        ]
        compact.pop("pages", None)
    if role == "derived_captions":
        compact = _compact_caption_inventory(compact)
    if role == "derived_body_geometry":
        compact = _compact_body_geometry(compact)
    if role == "derived_front_matter":
        compact = _compact_front_matter(compact)
    if role == "derived_abstract":
        compact = _compact_abstract(compact)
    if role == "derived_paragraph_metrics":
        compact = _compact_paragraph_metrics(compact)
    if role == "derived_headings":
        compact = _compact_headings(compact)
    if role == "derived_references":
        compact = {"heading": _compact_reference_heading(compact.get("heading"))}
    return compact


def _compact_body_geometry(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    return {
        key: source.get(key)
        for key in (
            "page_width_pt",
            "page_height_pt",
            "left_pt",
            "right_pt",
            "width_pt",
            "top_pt",
            "font_size_mode_pt",
            "font_name_mode",
            "font_name_times_compatible",
            "baseline_gap_median_pt",
            "sample_line_count",
            "sample_pages",
        )
    }


def _compact_front_matter(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    title = source.get("title") if isinstance(source.get("title"), dict) else {}
    if "horizontal_rules" in title:
        # ICML fused-layout schema. Keep the real parser fields instead of
        # applying the NeurIPS front-matter projection below.
        style = title.get("style") if isinstance(title.get("style"), dict) else {}
        title_bbox = title.get("bbox")
        rules = [item for item in title.get("horizontal_rules", []) if isinstance(item, dict)]
        rule_rows = []
        for item in rules:
            start, end = item.get("start"), item.get("end")
            if not (
                isinstance(start, list)
                and isinstance(end, list)
                and len(start) == len(end) == 2
                and all(isinstance(component, (int, float)) for component in start + end)
            ):
                continue
            rule_rows.append(
                {
                    "y_pt": round((float(start[1]) + float(end[1])) / 2, 4),
                    "width_pt": item.get("width_pt"),
                }
            )
        rule_rows.sort(key=lambda item: item["y_pt"])
        return {
            "schema": "icml_fused_layout",
            "title_bbox": title_bbox,
            "title_style": {
                "font_name": style.get("dominant_font"),
                "font_size_pt": style.get("font_size_median_pt"),
                "bold_ratio": style.get("bold_character_ratio"),
            },
            "title_alignment": title.get("alignment"),
            "horizontal_rule_count": len(rule_rows),
            "horizontal_rules": rule_rows,
            "top_rule_y_positions_pt": [rule_rows[0]["y_pt"]] if rule_rows else [],
            "bottom_rule_y_positions_pt": [rule_rows[-1]["y_pt"]] if rule_rows else [],
            "title_between_top_and_bottom_rules": bool(
                isinstance(title_bbox, list)
                and len(title_bbox) == 4
                and len(rule_rows) >= 2
                and rule_rows[0]["y_pt"] <= float(title_bbox[1])
                and rule_rows[-1]["y_pt"] >= float(title_bbox[3])
            ),
        }
    style = title.get("native_style") if isinstance(title.get("native_style"), dict) else {}
    title_bbox = source.get("title_bbox") or title.get("native_bbox") or title.get("bbox")
    top_rule_ys = _horizontal_rule_ys(source.get("top_rule_candidates"))
    bottom_rule_ys = _horizontal_rule_ys(source.get("bottom_rule_candidates"))
    return {
        "title_page_number": title.get("page_number"),
        "title_bbox": title_bbox,
        "title_style": {
            key: style.get(key)
            for key in ("font_name", "font_size_pt", "bold_ratio", "line_count")
        },
        "title_alignment_delta_pt": source.get("title_alignment_delta_pt"),
        "top_rule_count": len(source.get("top_rule_candidates") or []),
        "bottom_rule_count": len(source.get("bottom_rule_candidates") or []),
        "horizontal_rule_count": len(source.get("horizontal_rules") or []),
        "top_rule_y_positions_pt": top_rule_ys,
        "bottom_rule_y_positions_pt": bottom_rule_ys,
        "title_between_top_and_bottom_rules": (
            isinstance(title_bbox, list)
            and len(title_bbox) == 4
            and bool(top_rule_ys)
            and bool(bottom_rule_ys)
            and max(top_rule_ys) <= float(title_bbox[1])
            and min(bottom_rule_ys) >= float(title_bbox[3])
        ),
    }


def _horizontal_rule_ys(value: Any) -> list[float]:
    rows = value if isinstance(value, list) else []
    positions: list[float] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        box = item.get("bbox")
        if isinstance(box, list) and len(box) == 4 and all(
            isinstance(component, (int, float)) for component in box
        ):
            positions.append(round((float(box[1]) + float(box[3])) / 2, 4))
            continue
        start, end = item.get("start"), item.get("end")
        if (
            isinstance(start, list)
            and isinstance(end, list)
            and len(start) == len(end) == 2
            and isinstance(start[1], (int, float))
            and isinstance(end[1], (int, float))
        ):
            positions.append(round((float(start[1]) + float(end[1])) / 2, 4))
    return sorted(positions)


def _compact_abstract(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    body = source.get("body") if isinstance(source.get("body"), dict) else {}
    if body:
        # ICML stores all abstract-body measurements under `body`. Exposing
        # them as named top-level fields prevents a generic schema projection
        # from turning present parser measurements into null placeholders.
        heading = source.get("heading") if isinstance(source.get("heading"), dict) else {}
        heading_style = heading.get("style") if isinstance(heading.get("style"), dict) else {}
        body_style = body.get("style") if isinstance(body.get("style"), dict) else {}
        return {
            "schema": "icml_fused_layout",
            "heading_bbox": heading.get("bbox"),
            "heading_column_center_x_pt": heading.get("column_center_x_pt"),
            "heading_center_offset_from_column_pt": heading.get(
                "center_offset_from_column_pt"
            ),
            "heading_style": {
                "font_name": heading_style.get("dominant_font"),
                "font_size_pt": heading_style.get("font_size_median_pt"),
                "bold_ratio": heading_style.get("bold_character_ratio"),
            },
            "body_bbox": body.get("bbox"),
            "paragraph_count": body.get("paragraph_count"),
            "sentence_count": body.get("sentence_count"),
            "left_extra_indent_pt": body.get("left_extra_indent_pt"),
            "right_extra_indent_pt": body.get("right_extra_indent_pt"),
            "gap_after_pt": body.get("gap_after_pt"),
            "body_style": {
                "font_name": body_style.get("dominant_font"),
                "font_size_pt": body_style.get("font_size_median_pt"),
                "bold_ratio": body_style.get("bold_character_ratio"),
                "baseline_gap_pt": body_style.get("baseline_gap_median_pt"),
            },
        }
    style = source.get("style") if isinstance(source.get("style"), dict) else {}
    heading_style = source.get("heading_style") if isinstance(source.get("heading_style"), dict) else {}
    heading = source.get("heading") if isinstance(source.get("heading"), dict) else {}
    return {
        "heading_page_number": heading.get("page_number"),
        "heading_bbox": heading.get("native_bbox") or heading.get("bbox"),
        "heading_style": {
            key: heading_style.get(key)
            for key in ("font_name", "font_size_pt", "bold_ratio", "line_count")
        },
        "heading_alignment_delta_pt": source.get("heading_alignment_delta_pt"),
        "body_bbox": source.get("body_bbox"),
        "paragraph_count": source.get("paragraph_count"),
        "line_count": source.get("line_count"),
        "left_indent_pt": source.get("left_indent_pt"),
        "right_indent_pt": source.get("right_indent_pt"),
        "body_style": {
            key: style.get(key)
            for key in ("font_name", "font_size_pt", "bold_ratio", "baseline_gap_pt")
        },
    }


def _compact_paragraph_metrics(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    return {
        key: source.get(key)
        for key in (
            "paragraph_count_sampled",
            "first_line_indent_median_pt",
            "first_line_indent_abs_p90_pt",
            "paragraph_extra_gap_median_pt",
            "paragraph_extra_gap_samples",
        )
    }


def _compact_headings(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    rows = (
        source.get("items")
        if isinstance(source.get("items"), list)
        else value if isinstance(value, list) else []
    )
    by_depth: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in rows:
        if not isinstance(item, dict) or not isinstance(item.get("depth"), int):
            continue
        by_depth[item["depth"]].append(item)
    inventory: dict[str, Any] = {}
    for depth, items in sorted(by_depth.items()):
        styles = [item.get("style") for item in items if isinstance(item.get("style"), dict)]
        sizes = [item.get("font_size_pt") for item in styles if isinstance(item.get("font_size_pt"), (int, float))]
        bold = [item.get("bold_ratio") for item in styles if isinstance(item.get("bold_ratio"), (int, float))]
        offsets = [item.get("bbox", [None])[0] for item in items if isinstance(item.get("bbox"), list) and item["bbox"]]
        inventory[f"depth_{depth}"] = {
            "count": len(items),
            "pages": sorted({item.get("page_number") for item in items if isinstance(item.get("page_number"), int)}),
            "font_size_values_pt": sorted({round(float(item), 4) for item in sizes}),
            "bold_ratio_range": [round(min(bold), 4), round(max(bold), 4)] if bold else None,
            "left_x_values_pt": sorted({round(float(item), 4) for item in offsets}),
        }
    return {
        "levels": inventory,
        "title_case": source.get("title_case"),
    }


def _compact_reference_heading(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    style = value.get("native_style") if isinstance(value.get("native_style"), dict) else {}
    return {
        "page_number": value.get("page_number"),
        "text": value.get("text"),
        "bbox": value.get("native_bbox") or value.get("bbox"),
        "style": {
            key: style.get(key)
            for key in ("font_name", "font_size_pt", "bold_ratio", "line_count")
        },
    }


def _caption_label(text: str) -> str:
    match = re.match(r"\s*((?:figure|fig\.|table)\s*\d+)\b", text, re.IGNORECASE)
    return match.group(1) if match else ""


def _compact_caption_inventory(value: dict[str, Any]) -> dict[str, Any]:
    """Project full figure/table evidence without sending caption or OCR prose.

    The source inventory remains complete, but range/count aggregates are the
    minimum sufficient representation for numbering, caption style, and
    caption-pair direction rules.  Per-object geometry stays in the raw facts
    for rules that require a local anchor.
    """

    inventory: dict[str, Any] = {}
    for kind in ("figures", "tables"):
        source = value.get(kind)
        if not isinstance(source, dict):
            continue
        numbers: list[int] = []
        fonts: set[str] = set()
        font_sizes: list[float] = []
        paired_count = 0
        below_count = 0
        not_below_numbers: list[int] = []
        for item in source.get("items", []):
            if not isinstance(item, dict):
                continue
            style = item.get("caption_style") if isinstance(item.get("caption_style"), dict) else {}
            number = item.get("number")
            if isinstance(number, int):
                numbers.append(number)
            font = style.get("dominant_font")
            if isinstance(font, str) and font:
                fonts.add(font)
            size = style.get("font_size_median_pt")
            if isinstance(size, (int, float)):
                font_sizes.append(float(size))
            if item.get("caption_bbox") and item.get("paired_object_bbox"):
                paired_count += 1
            gap = item.get("gap_pt")
            if isinstance(gap, (int, float)):
                if gap >= 0:
                    below_count += 1
                elif isinstance(number, int):
                    not_below_numbers.append(number)
        inventory[kind] = {
            "count": source.get("count"),
            "numbers": source.get("numbers") or sorted(numbers),
            "numbering_continuous": source.get("numbering_continuous"),
            "caption_pair_count": paired_count,
            "caption_below_count": below_count,
            "caption_not_below_numbers": sorted(not_below_numbers),
            "caption_font_names": sorted(fonts),
            "caption_font_size_range_pt": (
                [round(min(font_sizes), 4), round(max(font_sizes), 4)]
                if font_sizes
                else None
            ),
        }
    return inventory


def _make_unverifiable(candidates: list[dict[str, Any]], reason: str) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for item in candidates:
        if item.get("result") in {"compliant", "non_compliant"}:
            converted.append({**item, "result": "unverifiable", "reason": reason, "evidence_status": "incomplete"})
        else:
            converted.append(item)
    return converted


def _unresolved_unit_rule_ids(
    findings: list[dict[str, Any]], expected_rule_ids: list[str]
) -> list[str]:
    """Identify rules without a complete terminal conclusion in one unit."""

    resolved: set[str] = set()
    for finding in findings:
        if finding.get("result") not in {"compliant", "non_compliant", "not_applicable"}:
            continue
        if finding.get("result") != "not_applicable" and finding.get("evidence_status") != "complete":
            continue
        resolved.update(str(rule_id) for rule_id in finding.get("rule_ids", []) if str(rule_id))
    return sorted(set(expected_rule_ids) - resolved)


def _freeze_complete_findings(
    findings: list[dict[str, Any]], unresolved_rule_ids: list[str]
) -> list[dict[str, Any]]:
    unresolved = set(unresolved_rule_ids)
    frozen: list[dict[str, Any]] = []
    for finding in findings:
        rule_ids = {str(rule_id) for rule_id in finding.get("rule_ids", []) if str(rule_id)}
        if (
            rule_ids
            and rule_ids.isdisjoint(unresolved)
            and finding.get("result") in {"compliant", "non_compliant", "not_applicable"}
            and (
                finding.get("result") == "not_applicable"
                or finding.get("evidence_status") == "complete"
            )
        ):
            frozen.append(finding)
    return frozen


def _force_rules_unverifiable(
    findings: list[dict[str, Any]], rule_ids: list[str], reason: str
) -> list[dict[str, Any]]:
    """Downgrade only rules whose requested recovery did not produce evidence."""

    target = set(rule_ids)
    forced: list[dict[str, Any]] = []
    for finding in findings:
        finding_rule_ids = {str(rule_id) for rule_id in finding.get("rule_ids", []) if str(rule_id)}
        if finding_rule_ids & target:
            forced.append(
                {
                    **finding,
                    "result": "unverifiable",
                    "severity": "info",
                    "suggestion": None,
                    "reason": reason,
                    "evidence_status": "incomplete",
                }
            )
        else:
            forced.append(finding)
    return forced


def _new_supplemental_facts(
    existing: list[dict[str, Any]], supplements: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    existing_keys = {
        (str(item.get("block_id") or ""), str(item.get("source_uri") or ""))
        for item in existing
    }
    return [
        item
        for item in supplements
        if (str(item.get("block_id") or ""), str(item.get("source_uri") or ""))
        not in existing_keys
    ]


def _new_standard_evidences(
    existing: list[dict[str, Any]], refreshed: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep only rule supplements that add source material, not a new ID."""

    existing_material = {_standard_evidence_material_key(item) for item in existing}
    return [
        item
        for item in refreshed
        if _standard_evidence_material_key(item) not in existing_material
    ]


def _standard_evidence_material_key(evidence: dict[str, Any]) -> tuple[str, str, str]:
    """Identify a standard by canonical rule and source text, never its transient S* ID."""

    return (
        str(evidence.get("canonical_rule_id") or ""),
        str(evidence.get("document_id") or ""),
        _text_key(str(evidence.get("quote") or "")),
    )


def _has_complete_actionable_finding(findings: list[dict[str, Any]]) -> bool:
    """Whether a unit contains a conclusion that survived evidence validation."""

    return any(
        item.get("result") in {"compliant", "non_compliant"}
        and item.get("evidence_status") == "complete"
        for item in findings
    )


def _annotations(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for finding in findings:
        for evidence in finding.get("paper_evidences", []):
            if _is_bbox(evidence.get("bbox")) and evidence.get("page_number") is not None:
                items.append(
                    {
                        "category": finding["category"],
                        "aspect": finding["aspect"],
                        "page": evidence["page_number"],
                        "bbox": evidence["bbox"],
                        "page_rotation": evidence.get("page_rotation", 0),
                        "block_id": evidence.get("block_id"),
                    }
                )
    return items


def _item_record(
    review_id: str,
    finding: dict[str, Any],
    *,
    unit_id: str | None = None,
    unit_position: int | None = None,
    source_stage: str = "final",
) -> FormatReviewItem:
    fingerprint = hashlib.sha256(
        f"{unit_id or 'final'}:{finding['category']}:{finding['aspect']}:{finding.get('result')}".encode()
    ).hexdigest()[:24]
    paper_refs = finding.get("paper_evidences", [])
    pages = sorted({int(item["page_number"]) for item in paper_refs if item.get("page_number") is not None})
    annotation = next((item for item in _annotations([finding])), {})
    canonical_rule_ids = [str(value) for value in finding.get("rule_ids", []) if str(value)]
    if source_stage == "final" and len(canonical_rule_ids) == 1:
        stored_rule_id = canonical_rule_ids[0]
    elif len(canonical_rule_ids) == 1 and unit_id:
        unit_fingerprint = hashlib.sha256(unit_id.encode()).hexdigest()[:16]
        stored_rule_id = f"{canonical_rule_ids[0]}:{unit_fingerprint}"
    else:
        stored_rule_id = f"{finding['category']}:{fingerprint}"
    return FormatReviewItem(
        format_review_id=review_id,
        unit_id=unit_id,
        unit_position=unit_position,
        source_stage=source_stage,
        rule_id=stored_rule_id,
        rule_title=str(finding["aspect"]),
        category=str(finding["category"]),
        aspect=str(finding["aspect"]),
        result=str(finding["result"]),
        severity=str(finding.get("severity") or "info"),
        evidence_status=str(finding.get("evidence_status") or "incomplete"),
        finding=str(finding["finding"]),
        suggestion=finding.get("suggestion"),
        page_numbers=pages,
        paper_evidence_json=paper_refs,
        standard_evidence_json=finding.get("standard_evidences", []),
        annotation_json=annotation,
    )


def _summary_markdown(findings: list[dict[str, Any]], coverage: dict[str, Any]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for item in findings:
        counts[str(item["result"])] += 1
    lines = ["## 格式审查摘要", "", f"- 不符合：{counts['non_compliant']} 项", f"- 无法可靠判断：{counts['unverifiable']} 项", f"- 符合：{counts['compliant']} 项"]
    if coverage.get("missing_categories"):
        lines.append(f"- 规范检索未完整覆盖：{'、'.join(coverage['missing_categories'])}")
    return "\n".join(lines)


def _font_is_bold(span: PdfTextSpanRecord) -> bool:
    """Use the native font flags/name; do not ask the model to infer weight."""

    font_name = " ".join(
        value for value in (span.raw_font_name, span.font_name) if isinstance(value, str)
    ).lower()
    return "bold" in font_name or bool((span.font_flags or 0) & 16)


def _matched_block_style(spans: list[PdfTextSpanRecord]) -> dict[str, Any] | None:
    """Aggregate a semantic block's matching native lines instead of picking one line."""

    if not spans:
        return None
    sizes = [float(span.font_size_pt) for span in spans if span.font_size_pt is not None]
    bboxes = [span.bbox_json for span in spans if _is_bbox(span.bbox_json)]
    primary = spans[0]
    return {
        "bbox": (
            [
                min(item[0] for item in bboxes),
                min(item[1] for item in bboxes),
                max(item[2] for item in bboxes),
                max(item[3] for item in bboxes),
            ]
            if bboxes
            else None
        ),
        "font_name": primary.font_name,
        "font_size_raw_pt": float(median(sizes)) if sizes else None,
        "font_size_pt": _nominal_font_size(float(median(sizes))) if sizes else None,
        "font_flags": primary.font_flags,
        "is_bold": any(_font_is_bold(span) for span in spans),
    }


def _nominal_font_size(value: float) -> float:
    """Map small PDF font-matrix drift to the author-facing nominal point size."""

    nearest = round(value)
    return float(nearest) if abs(value - nearest) <= 0.25 else value


def _model_unit_plan(unit: dict[str, Any]) -> dict[str, Any]:
    """Keep model metadata compact; the selected facts already carry evidence IDs."""

    return {
        key: unit.get(key)
        for key in (
            "unit_id",
            "unit_kind",
            "page_range",
        )
        if key in unit
    }


def _model_standard_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Expose only the rule text and IDs an LLM must cite."""

    compact = {
        key: evidence.get(key)
        for key in ("evidence_id", "canonical_rule_id", "category", "quote")
        if evidence.get(key) is not None
    }
    whitelist = _font_whitelist(str(evidence.get("quote") or ""))
    if not whitelist:
        whitelist = _font_attachment(str(evidence.get("source_attachment") or ""))
    if whitelist:
        compact["font_whitelist"] = whitelist
    return compact


def _font_whitelist(rule_text: str) -> list[str]:
    """Expose a manifest's embedded font whitelist as structured rule data."""

    match = re.search(r"字体白名单\s*[:：]\s*(.+)", rule_text, re.DOTALL)
    if match is None:
        return []
    values: list[str] = []
    for line in match.group(1).splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if not re.fullmatch(r"[A-Za-z0-9._-]+", candidate):
            break
        values.append(candidate)
    return values


def _font_attachment(value: str) -> list[str]:
    values = [line.strip() for line in value.splitlines() if line.strip()]
    # Font manifests use glob suffixes such as ``NimbusRomNo9L*``.  These are
    # rule data, not prose, and must survive projection into S*.font_whitelist.
    return values if values and all(re.fullmatch(r"[A-Za-z0-9.*_-]+", item) for item in values) else []


def _rule_review_policy(rule: dict[str, Any]) -> dict[str, Any]:
    """Choose the minimum auditable evidence scope for one atomic rule."""

    category = str(rule.get("rule_category") or "")
    selectors = {str(item) for item in rule.get("evidence_selector", [])}
    supported_checks = [str(item) for item in rule.get("supported_checks", []) if str(item)]
    configured_mode = str(rule.get("assessment_mode") or "").lower()
    is_global = bool(rule.get("is_global"))
    requires_cross_unit = bool(rule.get("requires_cross_unit"))

    if category in {"figure", "table"}:
        return {
            "mode": "complete_inventory",
            "supported_checks": supported_checks,
            "coverage_scope": "all_extracted_figures_and_tables",
            "selectors": sorted(selectors),
        }
    if is_global or requires_cross_unit or category in {
        "author_identity",
        "heading",
        "page_layout",
        "appendix",
        "reference",
    }:
        return {
            "mode": "sufficiency",
            "supported_checks": supported_checks,
            "coverage_scope": "derived_aggregate",
            "selectors": sorted(selectors),
        }
    if configured_mode == "sampled":
        return {
            "mode": "sampled",
            "supported_checks": supported_checks,
            "coverage_scope": "representative_sample",
            "selectors": sorted(selectors),
        }
    return {
        "mode": "tolerance_aware",
        "supported_checks": supported_checks,
        "coverage_scope": "localized",
        "selectors": sorted(selectors),
    }


def _horizontal_alignment(bbox: list[float] | None, page_width: float | None) -> str | None:
    if not _is_bbox(bbox) or not isinstance(page_width, (int, float)) or page_width <= 0:
        return None
    center = (float(bbox[0]) + float(bbox[2])) / 2
    tolerance = max(8.0, float(page_width) * 0.04)
    if abs(center - float(page_width) / 2) <= tolerance:
        return "center"
    if float(bbox[0]) <= float(page_width) * 0.12:
        return "left"
    if float(bbox[2]) >= float(page_width) * 0.88:
        return "right"
    return "indeterminate"


def _is_bbox(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 4 and all(isinstance(item, (int, float)) for item in value)


def _matching_spans(
    page_number: int,
    text: str,
    parsed_bbox: list[float] | None,
    spans: list[PdfTextSpanRecord],
) -> list[PdfTextSpanRecord]:
    normalized_text = _text_key(text)
    scored: list[tuple[int, PdfTextSpanRecord]] = []
    for span in spans:
        if span.page_number != page_number:
            continue
        span_text = _text_key(span.text)
        score = _span_text_match_score(normalized_text, span_text)
        if parsed_bbox is not None:
            overlap = bbox_iou(parsed_bbox, span.bbox_json)
            if overlap >= 0.8:
                score = max(score, 90 + int(overlap * 9))
        if score:
            scored.append((score, span))
    scored.sort(key=lambda item: (-item[0], item[1].span_index))
    return [span for _, span in scored]


def _span_text_match_score(block_text: str, span_text: str) -> int:
    """Return a conservative, deterministic parsed-block/native-span match score."""

    if not block_text or not span_text:
        return 0
    shortest = min(len(block_text), len(span_text))
    if shortest >= 8 and (span_text in block_text or block_text in span_text):
        return 100 + min(shortest, 50)
    block_tokens = set(re.findall(r"[a-z0-9]{3,}", block_text))
    span_tokens = set(re.findall(r"[a-z0-9]{3,}", span_text))
    if not block_tokens or not span_tokens:
        return 0
    shared = len(block_tokens & span_tokens)
    overlap = shared / min(len(block_tokens), len(span_tokens))
    if shared >= 3 and overlap >= 0.8:
        return 70 + int(overlap * 20)
    return 0


def _text_key(value: str) -> str:
    return "".join(value.split()).lower()


def _pdf_span_record_values(span: dict[str, Any]) -> dict[str, Any]:
    """Translate extractor geometry into the persisted PDF-facts schema."""

    values = dict(span)
    values["bbox_json"] = values.pop("bbox")
    return values


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ragflow_endpoint(base_url: str, resource: str) -> str:
    root = base_url.rstrip("/")
    if not root.endswith("/api/v1"):
        root = f"{root}/api/v1"
    return f"{root}/{resource.lstrip('/')}"
