"""Adapt the offline ``second_clean`` chunk rules for the API ingestion worker.

The user-paper pipeline owns the chunking rules.  This adapter deliberately
reuses those rules while keeping API uploads local: the resulting chunks are
persisted in PostgreSQL by ``IngestionTaskExecutor`` and are never imported
into a user RAGFlow dataset.
"""

from __future__ import annotations

import importlib
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SecondCleanResult:
    """Structured chunks and the non-fatal/fatal quality outcome."""

    chunks: list[dict[str, Any]]
    quality_report: dict[str, Any]
    blocking_errors: list[str]


@lru_cache
def _second_clean_module():
    """Load the maintained user-paper chunking implementation from this repo."""

    repository_root = Path(__file__).resolve().parents[3]
    source_root = repository_root / "user_paper"
    if not source_root.is_dir():
        raise RuntimeError("user_paper second_clean source is unavailable")
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)
    return importlib.import_module("second_clean")


def build_chunks(
    *,
    document: dict[str, Any],
    blocks: list[dict[str, Any]],
    media_objects: list[dict[str, Any]],
    ocr_by_id: dict[str, dict[str, Any]],
    target_tokens: int = 350,
    max_tokens: int = 550,
) -> SecondCleanResult:
    """Run second-clean's in-memory rules without writing RAGFlow artifacts."""

    cleaner = _second_clean_module()
    # The API adapter does not retain MinerU's raw content-list JSON.  Passing
    # a known-absent path still enables all repairs based on normalized blocks.
    repaired_blocks, repaired_media, repairs = cleaner.repair_mineru_blocks(
        Path("__api_second_clean_no_raw_mineru__"), blocks, media_objects
    )
    chunks = [cleaner.build_metadata_chunk(document, repaired_blocks)]
    chunks.extend(
        cleaner.group_text_blocks(
            document,
            repaired_blocks,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
        )
    )
    chunks.extend(cleaner.build_formula_chunks(document, repaired_blocks))
    chunks.extend(cleaner.build_algorithm_chunks(document, repaired_blocks))
    chunks.extend(cleaner.build_media_chunks(document, repaired_media, ocr_by_id))
    chunks.extend(cleaner.build_reference_chunks(document, repaired_blocks))
    cleaner.link_chunks(chunks)

    indexable = [chunk for chunk in chunks if chunk.get("indexable")]
    flags = Counter(flag for chunk in chunks for flag in chunk.get("quality_flags", []))
    critical: list[str] = []
    if not document.get("title"):
        critical.append("missing_title")
    if len(indexable) < 5:
        critical.append("too_few_indexable_chunks")
    if not any(chunk.get("content_role") == "abstract" for chunk in chunks):
        critical.append("missing_abstract_chunk")
    blocking_flags = {
        "unresolved_placeholder",
        "replacement_character",
        "missing_image",
        "table_content_missing",
        "ocr_failed",
    }
    has_blocking_flags = any(flag in blocking_flags for flag in flags)
    ocr_failures = sum(item.get("status") == "failed" for item in ocr_by_id.values())
    ocr_partials = sum(item.get("status") == "partial" for item in ocr_by_id.values())
    status = "failed" if critical else (
        "partial" if ocr_failures or ocr_partials or has_blocking_flags else "ready"
    )
    return SecondCleanResult(
        chunks=chunks,
        blocking_errors=critical,
        quality_report={
            "status": status,
            "critical_errors": critical,
            "indexable_chunks": len(indexable),
            "total_chunks": len(chunks),
            "parent_chunks": sum(not chunk.get("indexable") for chunk in chunks),
            "chunk_types": dict(Counter(chunk.get("content_type") for chunk in chunks)),
            "chunk_roles": dict(Counter(chunk.get("content_role") for chunk in chunks)),
            "quality_flags": dict(flags),
            "media_objects": len(repaired_media),
            "ocr_results": len(ocr_by_id),
            "ocr_failures": ocr_failures,
            "ocr_partials": ocr_partials,
            "mineru_repairs": repairs,
            "cleaning_version": cleaner.CLEANING_VERSION,
            "knowledge_base_import": "not_required",
        },
    )
