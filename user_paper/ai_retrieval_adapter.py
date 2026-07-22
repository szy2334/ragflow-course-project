"""Async RAGFlow adapter for the evidence contracts owned by ``app.ai``.

The ingestion scripts in this package create ``chunk_mapping.jsonl``.  This
module is the query-time counterpart: it resolves authorised paper IDs to
RAGFlow targets, retrieves chunks, and converts them into strict AI evidence.
It deliberately does not implement FastAPI, persistence, or public standards.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import httpx

from app.ai.ports import RetrievalPort
from app.ai.schemas import (
    EvidenceItem,
    EvidenceSet,
    RetrieveEvidenceRequest,
    RetrieveStandardsRequest,
)


class RagflowAdapterError(RuntimeError):
    """An invalid or unsuccessful RAGFlow retrieval operation."""


@dataclass(frozen=True, slots=True)
class RagflowRetrievalSettings:
    """Non-workflow settings for a RAGFlow retrieval endpoint."""

    base_url: str
    api_key: str = field(repr=False)
    timeout_seconds: float = 20.0
    retries: int = 1
    page_size: int = 8
    relaxed_page_size: int = 12
    similarity_threshold: float = 0.2
    relaxed_similarity_threshold: float = 0.0
    vector_similarity_weight: float = 0.3
    cross_languages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("base_url cannot be blank")
        if not self.api_key.strip():
            raise ValueError("api_key cannot be blank")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.retries < 0:
            raise ValueError("retries cannot be negative")
        if self.page_size <= 0 or self.relaxed_page_size <= 0:
            raise ValueError("page sizes must be positive")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if not 0 <= self.relaxed_similarity_threshold <= 1:
            raise ValueError("relaxed_similarity_threshold must be between 0 and 1")
        if not 0 <= self.vector_similarity_weight <= 1:
            raise ValueError("vector_similarity_weight must be between 0 and 1")

    @classmethod
    def from_environment(cls) -> "RagflowRetrievalSettings":
        """Load runtime-only settings without putting the API key in graph state."""

        api_key = os.environ.get("RAGFLOW_API_KEY")
        if not api_key:
            raise RagflowAdapterError("RAGFLOW_API_KEY is not set")
        languages = tuple(
            dict.fromkeys(
                part.strip()
                for part in os.environ.get("USER_PAPER_AI_CROSS_LANGUAGES", "").split(",")
                if part.strip()
            )
        )
        return cls(
            base_url=os.environ.get("RAGFLOW_BASE_URL", "http://localhost:9380/api/v1"),
            api_key=api_key,
            timeout_seconds=float(os.environ.get("USER_PAPER_AI_TIMEOUT", "20")),
            retries=int(os.environ.get("USER_PAPER_AI_RETRIES", "1")),
            page_size=int(os.environ.get("USER_PAPER_AI_PAGE_SIZE", "8")),
            relaxed_page_size=int(os.environ.get("USER_PAPER_AI_RELAXED_PAGE_SIZE", "12")),
            similarity_threshold=float(
                os.environ.get("USER_PAPER_AI_SIMILARITY_THRESHOLD", "0.2")
            ),
            relaxed_similarity_threshold=float(
                os.environ.get("USER_PAPER_AI_RELAXED_SIMILARITY_THRESHOLD", "0")
            ),
            vector_similarity_weight=float(
                os.environ.get("USER_PAPER_AI_VECTOR_SIMILARITY_WEIGHT", "0.3")
            ),
            cross_languages=languages,
        )


@dataclass(frozen=True, slots=True)
class PaperTarget:
    paper_id: str
    dataset_id: str
    document_id: str


class JsonlPaperRegistry:
    """Read authorised paper and chunk provenance from importer mapping files.

    A production database-backed registry can expose the same ``resolve`` and
    ``mapping_for_chunk`` methods.  Keeping that dependency behind this small
    object lets the AI workflow remain independent of database technology.
    """

    def __init__(
        self,
        mapping_paths: Sequence[Path | str],
        *,
        allow_legacy_unscoped_mappings: bool = False,
    ) -> None:
        self._allow_legacy_unscoped_mappings = allow_legacy_unscoped_mappings
        self._rows: list[dict[str, Any]] = []
        self._by_ragflow_chunk_id: dict[str, dict[str, Any]] = {}
        for mapping_path in mapping_paths:
            self._load(Path(mapping_path))

    def _load(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Chunk mapping does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected object at {path}:{line_number}")
                required = ("paper_id", "dataset_id", "document_id", "ragflow_chunk_id")
                if any(not str(row.get(field) or "").strip() for field in required):
                    raise ValueError(f"Incomplete RAGFlow mapping at {path}:{line_number}")
                chunk_id = str(row["ragflow_chunk_id"])
                if chunk_id in self._by_ragflow_chunk_id:
                    raise ValueError(f"Duplicate ragflow_chunk_id in mapping: {chunk_id}")
                self._rows.append(row)
                self._by_ragflow_chunk_id[chunk_id] = row

    def resolve(self, *, user_id: str, paper_ids: Iterable[str]) -> list[PaperTarget]:
        requested = set(paper_ids)
        targets: dict[tuple[str, str, str], PaperTarget] = {}
        for row in self._rows:
            if str(row["paper_id"]) not in requested or not self._is_authorised(row, user_id):
                continue
            target = PaperTarget(
                paper_id=str(row["paper_id"]),
                dataset_id=str(row["dataset_id"]),
                document_id=str(row["document_id"]),
            )
            targets[(target.paper_id, target.dataset_id, target.document_id)] = target
        return list(targets.values())

    def mapping_for_chunk(
        self,
        *,
        ragflow_chunk_id: str,
        user_id: str,
        paper_ids: Iterable[str],
    ) -> dict[str, Any] | None:
        row = self._by_ragflow_chunk_id.get(ragflow_chunk_id)
        if row is None:
            return None
        if str(row["paper_id"]) not in set(paper_ids) or not self._is_authorised(row, user_id):
            return None
        return row

    def _is_authorised(self, row: Mapping[str, Any], user_id: str) -> bool:
        mapped_user_id = str(row.get("user_id") or "").strip()
        return mapped_user_id == user_id or (
            not mapped_user_id and self._allow_legacy_unscoped_mappings
        )


class AsyncRagflowRetrievalClient:
    """Minimal async client for RAGFlow's retrieval endpoint."""

    def __init__(
        self,
        settings: RagflowRetrievalSettings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or httpx.AsyncClient(timeout=settings.timeout_seconds)
        self._owns_client = client is None

    async def retrieve(
        self,
        *,
        question: str,
        targets: Sequence[PaperTarget],
        relaxed: bool,
    ) -> dict[str, Any]:
        if not targets:
            raise ValueError("targets cannot be empty")
        body: dict[str, Any] = {
            "question": question,
            "dataset_ids": list(dict.fromkeys(target.dataset_id for target in targets)),
            "document_ids": list(dict.fromkeys(target.document_id for target in targets)),
            "page": 1,
            "page_size": (
                self._settings.relaxed_page_size if relaxed else self._settings.page_size
            ),
            "similarity_threshold": (
                self._settings.relaxed_similarity_threshold
                if relaxed
                else self._settings.similarity_threshold
            ),
            "vector_similarity_weight": self._settings.vector_similarity_weight,
        }
        if self._settings.cross_languages:
            body["cross_languages"] = list(self._settings.cross_languages)
        url = f"{self._settings.base_url.rstrip('/')}/retrieval"
        last_error: Exception | None = None
        for attempt in range(self._settings.retries + 1):
            try:
                response = await self._client.post(
                    url,
                    json=body,
                    headers={
                        "Authorization": f"Bearer {self._settings.api_key}",
                        "Accept": "application/json",
                    },
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise RagflowAdapterError(f"RAGFlow HTTP {response.status_code}")
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise RagflowAdapterError("RAGFlow returned a non-object JSON response")
                if payload.get("code") not in (None, 0, "0"):
                    raise RagflowAdapterError(
                        f"RAGFlow code {payload.get('code')}: {payload.get('message', '')}"
                    )
                return payload
            except (httpx.HTTPError, ValueError, RagflowAdapterError) as exc:
                last_error = exc
                if attempt < self._settings.retries:
                    await asyncio.sleep(min(2**attempt, 2))
        raise RagflowAdapterError(f"RAGFlow retrieval failed: {last_error}")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class RagflowRetrievalAdapter(RetrievalPort):
    """Adapt authorised RAGFlow paper chunks to the AI evidence contract."""

    def __init__(
        self,
        registry: JsonlPaperRegistry,
        client: AsyncRagflowRetrievalClient,
    ) -> None:
        self._registry = registry
        self._client = client

    async def retrieve_paper(self, request: RetrieveEvidenceRequest) -> EvidenceSet:
        targets = self._registry.resolve(user_id=request.user_id, paper_ids=request.paper_ids)
        if not targets:
            return EvidenceSet(
                items=[],
                query=request.standalone_question,
                relaxed=request.relaxed,
                warnings=["NO_AUTHORISED_PAPER_TARGETS"],
            )
        payload = await self._client.retrieve(
            question=request.standalone_question,
            targets=targets,
            relaxed=request.relaxed,
        )
        items, warnings = self._to_evidence_items(payload, request)
        return EvidenceSet(
            items=items,
            query=request.standalone_question,
            relaxed=request.relaxed,
            warnings=warnings,
        )

    async def retrieve_standards(self, request: RetrieveStandardsRequest) -> EvidenceSet:
        # Public standards are intentionally out of scope for the user-paper
        # pipeline.  The graph already knows how to degrade review/score flows
        # when no standard evidence is available.
        return EvidenceSet(
            items=[],
            query=request.standalone_question,
            warnings=["PUBLIC_STANDARD_RETRIEVAL_NOT_CONFIGURED"],
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _to_evidence_items(
        self,
        payload: Mapping[str, Any],
        request: RetrieveEvidenceRequest,
    ) -> tuple[list[EvidenceItem], list[str]]:
        items: list[EvidenceItem] = []
        warnings: list[str] = []
        seen_chunk_ids: set[str] = set()
        for raw_chunk in _response_chunks(payload):
            ragflow_chunk_id = str(raw_chunk.get("id") or raw_chunk.get("chunk_id") or "").strip()
            if not ragflow_chunk_id or ragflow_chunk_id in seen_chunk_ids:
                continue
            mapping = self._registry.mapping_for_chunk(
                ragflow_chunk_id=ragflow_chunk_id,
                user_id=request.user_id,
                paper_ids=request.paper_ids,
            )
            if mapping is None:
                warnings.append("DISCARDED_OUT_OF_SCOPE_RAGFLOW_CHUNK")
                continue
            quote = str(raw_chunk.get("content") or "").strip()
            if not quote:
                warnings.append("DISCARDED_EMPTY_RAGFLOW_CHUNK")
                continue
            seen_chunk_ids.add(ragflow_chunk_id)
            source_chunk_id = str(mapping.get("source_chunk_id") or ragflow_chunk_id)
            items.append(
                EvidenceItem(
                    evidence_id=f"P{len(items) + 1}",
                    source_type="paper",
                    paper_id=str(mapping["paper_id"]),
                    document_id=str(mapping["document_id"]),
                    chunk_id=ragflow_chunk_id,
                    content_type=_content_type(mapping.get("content_type")),
                    quote=quote,
                    section_title=_section_title(mapping),
                    page_number=_page_number(mapping.get("page_start")),
                    source_uri=f"paper://{mapping['paper_id']}/chunks/{source_chunk_id}",
                    retrieval_score=_retrieval_score(raw_chunk),
                    content_role=_optional_text(mapping.get("content_role")),
                    object_id=_optional_text(mapping.get("object_id")),
                    parent_chunk_id=_optional_text(mapping.get("parent_chunk_id")),
                    metadata={
                        "source_chunk_id": source_chunk_id,
                        "source_ref": mapping.get("source_ref"),
                        "section_path": mapping.get("section_path") or [],
                        "page_end": mapping.get("page_end"),
                        "quality_flags": mapping.get("quality_flags") or [],
                    },
                )
            )
        return items, list(dict.fromkeys(warnings))


def _response_chunks(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise RagflowAdapterError("RAGFlow retrieval response has no object data field")
    chunks = data.get("chunks")
    if not chunks:
        reference = data.get("reference")
        chunks = reference.get("chunks") if isinstance(reference, Mapping) else []
    return [chunk for chunk in chunks if isinstance(chunk, Mapping)] if isinstance(chunks, list) else []


def _content_type(value: Any) -> str:
    normalized = str(value or "text").strip().lower()
    aliases = {
        "abstract": "text",
        "body": "text",
        "title": "metadata",
        "chart": "figure",
        "image": "figure",
        "caption": "figure_caption",
        "reference_entry": "reference",
    }
    resolved = aliases.get(normalized, normalized)
    return resolved if resolved in {
        "text", "figure", "figure_caption", "table", "formula", "metadata", "reference"
    } else "text"


def _retrieval_score(chunk: Mapping[str, Any]) -> float:
    for score_field in ("retrieval_score", "similarity", "score", "vector_similarity"):
        value = chunk.get(score_field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0.0, float(value))
    return 0.0


def _section_title(mapping: Mapping[str, Any]) -> str | None:
    section = _optional_text(mapping.get("section"))
    if section:
        return section
    path = mapping.get("section_path")
    if isinstance(path, list):
        labels = [str(part).strip() for part in path if str(part).strip()]
        return " > ".join(labels) or None
    return None


def _page_number(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    return None


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
