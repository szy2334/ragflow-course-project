"""Adapters for the maintained MinerU and Baidu OCR user-paper pipeline.

The web ingestion worker must call the same provider contracts as the offline
pipeline.  The old lightweight endpoints (``/api/v1/parse`` and synthetic
``/table-recognition-v2``) are not MinerU/Baidu public APIs, so they made a
configured web upload fail even though the offline UMAP pipeline succeeded.

This module deliberately reuses the provider-specific source of truth in
``user_paper``.  It returns plain serialisable records; persistence
and ownership remain the responsibility of ``IngestionTaskExecutor``.
"""

from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import Settings


class ExternalPipelineError(RuntimeError):
    """A provider-stage error that is safe to expose as a generic task failure."""


@lru_cache
def _source_root() -> Path:
    root = Path(__file__).resolve().parents[3] / "user_paper"
    if not root.is_dir():
        raise ExternalPipelineError("user_paper pipeline source is unavailable")
    source = str(root)
    if source not in sys.path:
        sys.path.insert(0, source)
    return root


def _source_module(name: str):
    _source_root()
    return importlib.import_module(name)


def parse_mineru_pdf(
    settings: Settings,
    *,
    pdf_path: Path,
    paper_id: str,
    paper_version_id: str,
    artifact_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Upload a PDF to MinerU v4 and return the normalized block/media records."""

    if not settings.mineru_base_url or not settings.mineru_api_key:
        raise ExternalPipelineError("MinerU credentials are not configured")
    module = _source_module("mineru_clean")
    try:
        client = module.MineruClient(
            settings.mineru_api_key.get_secret_value(), settings.mineru_base_url, timeout=90.0
        )
        batch_id, upload_url = client.create_upload(pdf_path, paper_id)
        client.upload_file(upload_url, pdf_path)
        result_url = client.wait_result(batch_id, poll_seconds=5.0, max_wait_seconds=900.0)
        mineru_dir = client.download_and_extract(result_url, artifact_root / "raw_mineru")
        content_path = next(iter(sorted(mineru_dir.glob("*_content_list.json"))), None)
        if content_path is None:
            raise ExternalPipelineError("MinerU did not return a content-list artifact")
        content = module.read_json(content_path)
        if not isinstance(content, list):
            raise ExternalPipelineError("MinerU content-list artifact has an invalid format")
        blocks, media = module.build_blocks(
            content,
            paper_id=paper_id,
            paper_version_id=paper_version_id,
            mineru_dir=mineru_dir,
        )
        module.recover_references_from_markdown(
            blocks,
            markdown_path=mineru_dir / "full.md",
            paper_id=paper_id,
            paper_version_id=paper_version_id,
        )
    except ExternalPipelineError:
        raise
    except Exception as exc:  # Provider libraries deliberately use several exception types.
        raise ExternalPipelineError("MinerU parsing failed") from exc
    if not blocks:
        raise ExternalPipelineError("MinerU did not return usable blocks")
    return blocks, media


def recognize_baidu_media(
    settings: Settings,
    media: dict[str, Any],
    *,
    raw_root: Path,
) -> dict[str, Any]:
    """Run the strict Baidu table/PaddleOCR-VL enrichment used by second_clean."""

    if not settings.baidu_ocr_api_key or not settings.baidu_ocr_secret_key:
        raise ExternalPipelineError("Baidu OCR credentials are not configured")
    module = _source_module("baidu_ocr")
    # The reusable pipeline reads endpoint defaults on import.  Keep the web
    # runtime settings authoritative when an installation overrides them.
    module.TOKEN_URL = settings.baidu_ocr_token_url
    module.ACCURATE_URL = settings.baidu_ocr_accurate_url
    module.TABLE_URL = settings.baidu_ocr_table_url
    module.PADDLE_TASK_URL = settings.baidu_ocr_paddle_task_url
    module.PADDLE_QUERY_URL = settings.baidu_ocr_paddle_query_url
    try:
        client = module.BaiduOcrClient(
            settings.baidu_ocr_api_key.get_secret_value(),
            settings.baidu_ocr_secret_key.get_secret_value(),
            timeout=90.0,
            retries=3,
        )
        result = module.process_object(
            client,
            media,
            raw_dir=raw_root,
            skip_vl=False,
            poll_seconds=2.0,
            max_wait_seconds=300.0,
            strict_specialized=True,
        )
    except Exception as exc:
        raise ExternalPipelineError("Baidu specialized OCR failed") from exc
    if result.get("status") != "success":
        raise ExternalPipelineError("Baidu specialized OCR returned no usable result")
    return result
