"""Acquire a MinerU result and convert it into stable business-side blocks.

The script accepts an existing MinerU export directory or submits a PDF to the
MinerU Open Platform. Raw MinerU output is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import requests

from pipeline_common import (
    PARSER_PIPELINE_VERSION,
    PipelineError,
    config_value,
    normalize_prose,
    normalize_text,
    quality_flags,
    read_json,
    require_env,
    sanitize_name,
    sha256_file,
    stable_uuid,
    utc_now,
    write_json,
    write_jsonl,
)


DEFAULT_MINERU_BASE_URL = "https://mineru.net/api/v4"


class MineruClient:
    def __init__(self, token: str, base_url: str, timeout: float = 90.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        # MinerU traffic is direct; bypass a machine-level proxy that can close API polls.
        self.session.trust_env = False
        self.session.headers.update(
            {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        )

    @staticmethod
    def _payload(response: requests.Response) -> dict[str, Any]:
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") not in (None, 0, "0"):
            raise PipelineError(
                f"MinerU API error {payload.get('code')}: "
                f"{payload.get('msg') or payload.get('message') or payload}"
            )
        return payload

    def create_upload(self, pdf_path: Path, data_id: str) -> tuple[str, str]:
        body = {
            "files": [
                {
                    "name": pdf_path.name,
                    "data_id": data_id,
                    "is_ocr": True,
                }
            ],
            "model_version": "vlm",
            "enable_formula": True,
            "enable_table": True,
        }
        response = self.session.post(
            f"{self.base_url}/file-urls/batch", json=body, timeout=self.timeout
        )
        payload = self._payload(response)
        data = payload.get("data") or {}
        batch_id = data.get("batch_id")
        urls = data.get("file_urls") or data.get("upload_urls") or []
        if not batch_id or not urls:
            raise PipelineError(f"MinerU did not return batch_id/file_urls: {payload}")
        first = urls[0]
        if isinstance(first, dict):
            upload_url = first.get("url") or first.get("file_url")
        else:
            upload_url = first
        if not upload_url:
            raise PipelineError(f"MinerU did not return an upload URL: {payload}")
        return str(batch_id), str(upload_url)

    def upload_file(self, upload_url: str, pdf_path: Path) -> None:
        with pdf_path.open("rb") as handle:
            response = self.session.put(upload_url, data=handle, timeout=self.timeout)
        response.raise_for_status()

    def wait_result(
        self, batch_id: str, *, poll_seconds: float = 5.0, max_wait_seconds: float = 900
    ) -> str:
        deadline = time.time() + max_wait_seconds
        last_state = "unknown"
        while time.time() < deadline:
            response = self.session.get(
                f"{self.base_url}/extract-results/batch/{batch_id}",
                timeout=self.timeout,
            )
            payload = self._payload(response)
            data = payload.get("data") or {}
            results = (
                data.get("extract_result")
                or data.get("extract_results")
                or data.get("files")
                or []
            )
            if not results:
                last_state = str(data.get("state") or "waiting")
                time.sleep(poll_seconds)
                continue
            result = results[0]
            state = str(result.get("state") or result.get("status") or "").lower()
            last_state = state
            if state in {"done", "success", "completed"}:
                url = (
                    result.get("full_zip_url")
                    or result.get("zip_url")
                    or result.get("result_url")
                )
                if not url:
                    raise PipelineError(f"MinerU completed without a ZIP URL: {result}")
                return str(url)
            if state in {"failed", "error"}:
                raise PipelineError(
                    f"MinerU extraction failed: "
                    f"{result.get('err_msg') or result.get('error') or result}"
                )
            time.sleep(poll_seconds)
        raise TimeoutError(f"MinerU extraction timed out; last state={last_state}")

    def download_and_extract(self, zip_url: str, target_dir: Path) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        zip_path = target_dir / "mineru_result.zip"
        with self.session.get(zip_url, stream=True, timeout=self.timeout) as response:
            response.raise_for_status()
            with zip_path.open("wb") as handle:
                for block in response.iter_content(1024 * 1024):
                    if block:
                        handle.write(block)
        extract_dir = target_dir / "extracted"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                resolved = (extract_dir / member.filename).resolve()
                if extract_dir.resolve() not in resolved.parents and resolved != extract_dir.resolve():
                    raise PipelineError(f"Unsafe path in MinerU ZIP: {member.filename}")
            archive.extractall(extract_dir)
        return find_mineru_export(extract_dir)


def find_mineru_export(root: Path) -> Path:
    candidates: list[Path] = []
    if root.is_dir():
        candidates.append(root)
        candidates.extend(path.parent for path in root.rglob("*_content_list.json"))
    for candidate in candidates:
        if list(candidate.glob("*_content_list.json")):
            return candidate
    raise PipelineError(f"No *_content_list.json found under {root}")


def acquire_mineru_export(args: argparse.Namespace, paper_id: str) -> Path:
    if args.mineru_dir:
        return find_mineru_export(args.mineru_dir)
    try:
        token = require_env("MINERU_TOKEN")
    except PipelineError as exc:
        raise PipelineError(
            "MINERU_TOKEN is not configured and --mineru-dir was not provided"
        ) from exc
    raw_dir = args.output_dir / "raw_mineru"
    client = MineruClient(token, args.mineru_base_url, timeout=args.timeout)
    batch_id, upload_url = client.create_upload(args.pdf, paper_id)
    write_json(
        raw_dir / "submission.json",
        {
            "batch_id": batch_id,
            "file_name": args.pdf.name,
            "submitted_at": utc_now(),
            "base_url": args.mineru_base_url,
        },
    )
    client.upload_file(upload_url, args.pdf)
    result_url = client.wait_result(
        batch_id,
        poll_seconds=args.poll_seconds,
        max_wait_seconds=args.max_wait_seconds,
    )
    # The signed result URL is intentionally not persisted.
    return client.download_and_extract(result_url, raw_dir)


def caption_for(item: dict[str, Any]) -> list[str]:
    for key in ("image_caption", "chart_caption", "table_caption"):
        value = item.get(key)
        if isinstance(value, list):
            return [normalize_prose(str(part)) for part in value if str(part).strip()]
        if value:
            return [normalize_prose(str(value))]
    return []


def build_blocks(
    content: list[dict[str, Any]],
    *,
    paper_id: str,
    paper_version_id: str,
    mineru_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocks: list[dict[str, Any]] = []
    media: list[dict[str, Any]] = []
    heading_stack: list[str] = []

    for source_index, item in enumerate(content):
        item_type = str(item.get("type") or "unknown")
        page = int(item.get("page_idx", 0)) + 1
        bbox = item.get("bbox")
        raw_text = str(item.get("text") or "")
        captions = caption_for(item)
        normalized = (
            normalize_text(raw_text)
            if item_type == "equation"
            else normalize_prose(raw_text)
        )

        if item_type == "text" and item.get("text_level") is not None:
            level = max(1, int(item.get("text_level") or 1))
            heading_stack = heading_stack[: level - 1]
            while len(heading_stack) < level - 1:
                heading_stack.append("")
            heading_stack.append(normalized)
            role = "section_heading"
            content_type = "heading"
        elif item_type == "ref_text":
            role = "reference_entry"
            content_type = "text"
        elif item_type == "equation":
            role = "display_formula"
            content_type = "formula"
        elif item_type in {"image", "chart"}:
            role = "figure" if item_type == "image" else "chart"
            content_type = item_type
        elif item_type == "table":
            role = "table"
            content_type = "table"
        elif item_type == "page_footnote":
            role = "page_footnote"
            content_type = "text"
        elif item_type in {"header", "page_number"}:
            # Preserve an audit record but do not make these indexable.
            role = item_type
            content_type = "navigation"
        else:
            role = "paragraph"
            content_type = "text"

        section_path = [part for part in heading_stack if part]
        source_ref = f"paper://{paper_id}/page/{page}/block/{source_index}"
        block_id = stable_uuid(
            f"{paper_version_id}:block:{source_index}:{item_type}:{page}"
        )
        flags = quality_flags(normalized, require_text=content_type in {"text", "heading"})
        indexable = item_type not in {"header", "page_number", "page_footnote"}

        block = {
            "schema_version": "paper_block_v1",
            "block_id": block_id,
            "paper_id": paper_id,
            "paper_version_id": paper_version_id,
            "source_index": source_index,
            "content_type": content_type,
            "content_role": role,
            "raw_text": raw_text,
            "normalized_text": normalized,
            "caption": captions,
            "section_path": section_path,
            "page_start": page,
            "page_end": page,
            "bbox": bbox,
            "source_ref": source_ref,
            "indexable": indexable,
            "quality_flags": flags,
        }
        blocks.append(block)

        if item_type in {"image", "chart", "table"}:
            image_relative = item.get("img_path")
            image_path = mineru_dir / str(image_relative) if image_relative else None
            object_id = stable_uuid(
                f"{paper_version_id}:media:{source_index}:{item_type}:{page}"
            )
            media_flags = list(flags)
            if not captions:
                media_flags.append("missing_caption")
            if not image_path or not image_path.exists():
                media_flags.append("missing_image")
            media.append(
                {
                    "schema_version": "paper_media_v1",
                    "object_id": object_id,
                    "block_id": block_id,
                    "paper_id": paper_id,
                    "paper_version_id": paper_version_id,
                    "object_type": item_type,
                    "caption": captions,
                    "footnote": item.get(f"{item_type}_footnote") or [],
                    "mineru_content": normalize_prose(str(item.get("content") or "")),
                    "table_html": str(item.get("table_body") or ""),
                    "image_path": str(image_path.resolve()) if image_path else None,
                    "section_path": section_path,
                    "page_start": page,
                    "page_end": page,
                    "pdf_bbox": bbox,
                    "source_ref": source_ref.replace("/block/", "/object/"),
                    "quality_flags": sorted(set(media_flags)),
                }
            )

    text_positions = [
        index
        for index, block in enumerate(blocks)
        if block["content_role"] == "paragraph"
        and block["normalized_text"]
    ]
    for obj in media:
        block_index = next(
            index for index, block in enumerate(blocks) if block["block_id"] == obj["block_id"]
        )
        nearby = sorted(text_positions, key=lambda index: abs(index - block_index))[:4]
        nearby.sort()
        obj["nearby_block_ids"] = [blocks[index]["block_id"] for index in nearby]
        obj["nearby_text"] = [blocks[index]["normalized_text"] for index in nearby]
    return blocks, media


def recover_references_from_markdown(
    blocks: list[dict[str, Any]],
    *,
    markdown_path: Path,
    paper_id: str,
    paper_version_id: str,
) -> int:
    if any(block.get("content_role") == "reference_entry" for block in blocks):
        return 0
    if not markdown_path.exists():
        return 0
    markdown = markdown_path.read_text(encoding="utf-8")
    match = re.search(r"(?im)^##\s+References\s*$", markdown)
    if not match:
        return 0
    reference_text = markdown[match.end() :]
    entries = re.findall(
        r"(?ms)^\[(\d+)\]\s+(.*?)(?=^\[\d+\]\s+|\Z)", reference_text
    )
    page = max((int(block.get("page_end") or 0) for block in blocks), default=0) or None
    recovered = 0
    for number, value in entries:
        text = normalize_prose(f"[{number}] {value}")
        if not text:
            continue
        source_index = len(blocks)
        source_ref = f"paper://{paper_id}/section/references/{number}"
        blocks.append(
            {
                "schema_version": "paper_block_v1",
                "block_id": stable_uuid(
                    f"{paper_version_id}:markdown-reference:{number}:{text}"
                ),
                "paper_id": paper_id,
                "paper_version_id": paper_version_id,
                "source_index": source_index,
                "content_type": "text",
                "content_role": "reference_entry",
                "raw_text": text,
                "normalized_text": text,
                "caption": [],
                "section_path": ["References"],
                "page_start": page,
                "page_end": page,
                "bbox": None,
                "source_ref": source_ref,
                "indexable": True,
                "quality_flags": [],
                "locator_type": "markdown_reference_fallback",
            }
        )
        recovered += 1
    return recovered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mineru-dir", type=Path)
    parser.add_argument(
        "--mineru-base-url",
        default=config_value("MINERU_BASE_URL", DEFAULT_MINERU_BASE_URL),
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--max-wait-seconds", type=float, default=900.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.pdf = args.pdf.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    file_sha256 = sha256_file(args.pdf)
    paper_id = stable_uuid(f"paper:{file_sha256}")
    mineru_dir = acquire_mineru_export(args, paper_id)

    content_path = next(iter(sorted(mineru_dir.glob("*_content_list.json"))), None)
    if content_path is None:
        raise PipelineError(f"MinerU content list is missing in {mineru_dir}")
    content = read_json(content_path)
    if not isinstance(content, list):
        raise PipelineError(f"Unexpected MinerU content list format: {content_path}")

    layout_path = mineru_dir / "layout.json"
    layout_meta: dict[str, Any] = {}
    if layout_path.exists():
        layout = read_json(layout_path)
        if isinstance(layout, dict):
            layout_meta = {
                "backend": layout.get("_backend"),
                "effort": layout.get("_effort"),
                "ocr_enabled": layout.get("_ocr_enable"),
                "version": layout.get("_version_name"),
            }
    parser_version = str(layout_meta.get("version") or "mineru_unknown")
    paper_version_id = stable_uuid(
        f"paper-version:{file_sha256}:{parser_version}:{PARSER_PIPELINE_VERSION}"
    )
    blocks, media = build_blocks(
        content,
        paper_id=paper_id,
        paper_version_id=paper_version_id,
        mineru_dir=mineru_dir,
    )
    recovered_references = recover_references_from_markdown(
        blocks,
        markdown_path=mineru_dir / "full.md",
        paper_id=paper_id,
        paper_version_id=paper_version_id,
    )

    title = next(
        (
            block["normalized_text"]
            for block in blocks
            if block["content_role"] == "section_heading"
            and block["normalized_text"]
        ),
        args.pdf.stem,
    )
    document = {
        "schema_version": "paper_document_v1",
        "paper_id": paper_id,
        "paper_version_id": paper_version_id,
        "file_name": args.pdf.name,
        "file_path": str(args.pdf),
        "file_sha256": file_sha256,
        "title": title.lstrip("# "),
        "parser_name": "mineru",
        "parser_version": parser_version,
        "cleaning_version": PARSER_PIPELINE_VERSION,
        "mineru_export_dir": str(mineru_dir.resolve()),
        "mineru_metadata": layout_meta,
        "block_count": len(blocks),
        "media_count": len(media),
        "recovered_references": recovered_references,
        "created_at": utc_now(),
    }
    write_json(args.output_dir / "document.json", document)
    write_jsonl(args.output_dir / "blocks.jsonl", blocks)
    write_jsonl(args.output_dir / "media_objects.jsonl", media)
    write_json(
        args.output_dir / "mineru_clean_summary.json",
        {
            "paper_id": paper_id,
            "paper_version_id": paper_version_id,
            "document_title": document["title"],
            "blocks": len(blocks),
            "media_objects": len(media),
            "recovered_references": recovered_references,
            "quality_flag_counts": {
                flag: sum(flag in block["quality_flags"] for block in blocks)
                for flag in sorted(
                    {flag for block in blocks for flag in block["quality_flags"]}
                )
            },
            "output_dir": str(args.output_dir),
        },
    )
    print(
        json.dumps(
            {
                "stage": "mineru_clean",
                "paper_id": paper_id,
                "blocks": len(blocks),
                "media_objects": len(media),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
