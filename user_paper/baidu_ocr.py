"""Enrich MinerU media objects with Baidu OCR and PaddleOCR-VL results."""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

import requests

from pipeline_common import (
    PipelineError,
    config_value,
    iter_jsonl,
    normalize_prose,
    recursive_values,
    require_env,
    utc_now,
    write_json,
    write_jsonl,
)


TOKEN_URL = config_value(
    "BAIDU_OCR_TOKEN_URL", "https://aip.baidubce.com/oauth/2.0/token"
)
ACCURATE_URL = config_value(
    "BAIDU_OCR_ACCURATE_URL",
    "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic",
)
TABLE_URL = config_value("BAIDU_OCR_TABLE_URL", "https://aip.baidubce.com/rest/2.0/ocr/v1/table")
PADDLE_TASK_URL = (
    config_value(
        "BAIDU_OCR_PADDLE_TASK_URL",
        "https://aip.baidubce.com/rest/2.0/brain/online/v2/paddle-vl-parser/task",
    )
)
PADDLE_QUERY_URL = (
    config_value(
        "BAIDU_OCR_PADDLE_QUERY_URL",
        "https://aip.baidubce.com/rest/2.0/brain/online/v2/paddle-vl-parser/task/query",
    )
)


class BaiduOcrClient:
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        timeout: float = 90.0,
        retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self._token: str | None = None
        self._token_expires_at = 0.0

    def access_token(self) -> str:
        if self._token and time.time() < self._token_expires_at:
            return self._token
        response = self.session.get(
            TOKEN_URL,
            params={
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.secret_key,
            },
            timeout=self.timeout,
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise PipelineError(
                f"Baidu OCR authentication returned HTTP {response.status_code} "
                "with a non-JSON response"
            ) from exc
        if response.status_code >= 400:
            raise PipelineError(
                f"Baidu OCR authentication returned HTTP {response.status_code}: "
                f"{payload.get('error_description') or payload.get('error') or 'unauthorized'}"
            )
        token = payload.get("access_token")
        if not token:
            raise PipelineError(
                f"Baidu OCR authentication failed: "
                f"{payload.get('error_description') or payload.get('error') or payload}"
            )
        expires_in = int(payload.get("expires_in") or 3600)
        self._token = str(token)
        self._token_expires_at = time.time() + max(60, expires_in - 300)
        return self._token

    def _post_form(self, url: str, data: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.session.post(
                    url,
                    params={"access_token": self.access_token()},
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                if payload.get("error_code") not in (None, 0, "0"):
                    error_code = payload.get("error_code")
                    if error_code in {110, 111} and attempt < self.retries:
                        self._token = None
                        continue
                    raise PipelineError(
                        f"Baidu OCR API error {error_code}: "
                        f"{payload.get('error_msg') or payload}"
                    )
                return payload
            except (requests.RequestException, ValueError, PipelineError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(2**attempt, 8))
        if isinstance(last_error, PipelineError):
            raise last_error
        raise PipelineError(
            f"Baidu OCR request failed: {type(last_error).__name__ if last_error else 'unknown'}"
        )

    @staticmethod
    def _encoded_file(path: Path) -> str:
        return base64.b64encode(path.read_bytes()).decode("ascii")

    def accurate(self, image_path: Path) -> dict[str, Any]:
        return self._post_form(
            ACCURATE_URL,
            {
                "image": self._encoded_file(image_path),
                "detect_direction": "true",
                "probability": "true",
            },
        )

    def table(self, image_path: Path) -> dict[str, Any]:
        return self._post_form(
            TABLE_URL,
            {"image": self._encoded_file(image_path)},
        )

    def submit_paddle_vl(self, image_path: Path, *, analysis_chart: bool) -> str:
        payload = self._post_form(
            PADDLE_TASK_URL,
            {
                "file_data": self._encoded_file(image_path),
                "file_name": image_path.name,
                "analysis_chart": "true" if analysis_chart else "false",
            },
        )
        task_ids = recursive_values(payload, "task_id")
        if not task_ids:
            raise PipelineError(f"PaddleOCR-VL did not return task_id: {payload}")
        return str(task_ids[0])

    def query_paddle_vl(self, task_id: str) -> dict[str, Any]:
        return self._post_form(PADDLE_QUERY_URL, {"task_id": task_id})

    def wait_paddle_vl(
        self,
        task_id: str,
        *,
        poll_seconds: float = 2.0,
        max_wait_seconds: float = 300.0,
    ) -> dict[str, Any]:
        deadline = time.time() + max_wait_seconds
        last_status = "unknown"
        while time.time() < deadline:
            payload = self.query_paddle_vl(task_id)
            statuses = recursive_values(payload, "status")
            status = str(statuses[0] if statuses else "").lower()
            last_status = status or "unknown"
            if status == "success":
                return payload
            if status == "failed":
                errors = recursive_values(payload, "task_error")
                raise PipelineError(
                    f"PaddleOCR-VL task failed: {errors[0] if errors else payload}"
                )
            time.sleep(poll_seconds)
        raise TimeoutError(f"PaddleOCR-VL timed out; last status={last_status}")

    def download_result_url(self, url: str) -> tuple[str, Any]:
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "json" in content_type or response.text.lstrip().startswith(("{", "[")):
            return "json", response.json()
        return "text", response.text


def normalize_accurate(payload: dict[str, Any]) -> dict[str, Any]:
    lines: list[dict[str, Any]] = []
    confidences: list[float] = []
    for row in payload.get("words_result") or []:
        if not isinstance(row, dict):
            continue
        probability = row.get("probability") or {}
        average = probability.get("average") if isinstance(probability, dict) else None
        if isinstance(average, (int, float)):
            confidences.append(float(average))
        lines.append(
            {
                "text": normalize_prose(str(row.get("words") or "")),
                "location": row.get("location"),
                "confidence": average,
            }
        )
    return {
        "ocr_lines": lines,
        "ocr_text": "\n".join(row["text"] for row in lines if row["text"]),
        "ocr_average_confidence": (
            sum(confidences) / len(confidences) if confidences else None
        ),
    }


def normalized_string_values(payload: Any, keys: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for key in keys:
        for value in recursive_values(payload, key):
            if isinstance(value, str) and value.strip():
                candidate = value.strip()
                if candidate not in values:
                    values.append(candidate)
    return values


def normalize_table(payload: dict[str, Any]) -> dict[str, Any]:
    matrices: list[list[list[str]]] = []
    for table in payload.get("tables_result") or []:
        if not isinstance(table, dict):
            continue
        cells = []
        for key in ("header", "body", "footer"):
            value = table.get(key) or []
            if isinstance(value, list):
                cells.extend(cell for cell in value if isinstance(cell, dict))
        max_row = max((int(cell.get("row_end") or 0) for cell in cells), default=0)
        max_col = max((int(cell.get("col_end") or 0) for cell in cells), default=0)
        if not cells or max_row <= 0 or max_col <= 0:
            continue
        matrix = [["" for _ in range(max_col)] for _ in range(max_row)]
        for cell in cells:
            row_start = int(cell.get("row_start") or 0)
            row_end = max(row_start + 1, int(cell.get("row_end") or row_start + 1))
            col_start = int(cell.get("col_start") or 0)
            col_end = max(col_start + 1, int(cell.get("col_end") or col_start + 1))
            words = normalize_prose(str(cell.get("words") or ""))
            # Preserve merged-cell semantics. Repeating the same label in every
            # covered coordinate makes scientific multi-level headers unusable.
            if row_start < max_row and col_start < max_col:
                matrix[row_start][col_start] = words
        matrices.append(matrix)

    markdown_candidates: list[str] = []
    for matrix in matrices:
        if not matrix:
            continue
        width = max(len(row) for row in matrix)
        rows = [row + [""] * (width - len(row)) for row in matrix]
        rendered = ["| " + " | ".join(value.replace("|", "\\|") for value in rows[0]) + " |"]
        rendered.append("| " + " | ".join("---" for _ in range(width)) + " |")
        rendered.extend(
            "| " + " | ".join(value.replace("|", "\\|") for value in row) + " |"
            for row in rows[1:]
        )
        markdown_candidates.append("\n".join(rendered))

    return {
        "table_html_candidates": normalized_string_values(
            payload, ("table_html", "html", "body")
        ),
        "table_markdown_candidates": list(
            dict.fromkeys(
                markdown_candidates
                + normalized_string_values(payload, ("markdown", "table_markdown"))
            )
        ),
        "table_matrices": matrices,
        "table_cells": recursive_values(payload, "cells"),
    }


def normalize_paddle(payload: dict[str, Any], downloaded: list[Any]) -> dict[str, Any]:
    sources: list[Any] = [payload, *downloaded]
    descriptions: list[str] = []
    markdown_values: list[str] = []
    formula_values: list[str] = []
    for source in sources:
        descriptions.extend(
            normalized_string_values(source, ("image_description", "chart_description"))
        )
        markdown_values.extend(normalized_string_values(source, ("markdown",)))
        formula_values.extend(
            normalized_string_values(source, ("latex", "formula", "formula_text"))
        )
        for container in _dicts(source):
            if str(container.get("type") or "").lower() == "chart":
                chart_text = normalize_prose(str(container.get("text") or ""))
                if chart_text:
                    descriptions.append(chart_text)
    return {
        "visual_descriptions": list(dict.fromkeys(descriptions)),
        "paddle_markdown": list(dict.fromkeys(markdown_values)),
        "formula_latex": list(dict.fromkeys(formula_values)),
    }


def redact_downloaded_content(value: Any) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, child in value.items():
            if key in {"data_url", "markdown_url", "parse_result_url"} and isinstance(child, str):
                output[key] = "<temporary-url-redacted>"
            else:
                output[key] = redact_downloaded_content(child)
        return output
    if isinstance(value, list):
        return [redact_downloaded_content(child) for child in value]
    if isinstance(value, str):
        return re.sub(
            r"https?://[^\s\"'<>]+(?:authorization|x-bce-signature)[^\s\"'<>]*",
            "<temporary-url-redacted>",
            value,
            flags=re.IGNORECASE,
        )
    return value


def process_object(
    client: BaiduOcrClient,
    obj: dict[str, Any],
    *,
    raw_dir: Path,
    skip_vl: bool,
    poll_seconds: float,
    max_wait_seconds: float,
    strict_specialized: bool,
) -> dict[str, Any]:
    object_id = str(obj["object_id"])
    image_path_value = obj.get("image_path")
    image_path = Path(str(image_path_value)) if image_path_value else None
    if not image_path or not image_path.exists():
        return {
            "object_id": object_id,
            "status": "skipped",
            "error": "image_not_found",
            "quality_flags": ["missing_image"],
        }

    raw_object_dir = raw_dir / object_id
    raw_object_dir.mkdir(parents=True, exist_ok=True)
    object_type = str(obj.get("object_type") or "image")
    result: dict[str, Any] = {
        "schema_version": "paper_ocr_v1",
        "object_id": object_id,
        "paper_id": obj.get("paper_id"),
        "paper_version_id": obj.get("paper_version_id"),
        "object_type": object_type,
        "status": "success",
        "processors": [],
        "processed_at": utc_now(),
        "quality_flags": [],
        "errors": [],
    }

    if object_type == "table":
        try:
            payload = client.table(image_path)
            write_json(raw_object_dir / "baidu_table.json", payload)
            result["processors"].append("baidu_table_v2")
            result.update(normalize_table(payload))
        except Exception as exc:
            if strict_specialized:
                raise
            # Keep the paper usable when the account has not enabled the paid
            # table API. MinerU table HTML remains the structure source.
            accurate = client.accurate(image_path)
            write_json(raw_object_dir / "baidu_table_accurate_fallback.json", accurate)
            result["processors"].append("baidu_accurate_basic_table_fallback")
            result.update(normalize_accurate(accurate))
            result["status"] = "partial"
            result["errors"].append(str(exc))
            result["quality_flags"].append("table_structure_ocr_failed")
    else:
        accurate = client.accurate(image_path)
        write_json(raw_object_dir / "baidu_accurate.json", accurate)
        result["processors"].append("baidu_accurate_basic")
        result.update(normalize_accurate(accurate))
        confidence = result.get("ocr_average_confidence")
        if isinstance(confidence, (int, float)) and confidence < 0.75:
            result["quality_flags"].append("ocr_low_confidence")

        if not skip_vl:
            try:
                task_id = client.submit_paddle_vl(image_path, analysis_chart=True)
                paddle_payload = client.wait_paddle_vl(
                    task_id,
                    poll_seconds=poll_seconds,
                    max_wait_seconds=max_wait_seconds,
                )
                # Persist task_id for audit; signed URLs and access tokens are not persisted.
                redacted_payload = json.loads(json.dumps(paddle_payload))
                for key in ("markdown_url", "parse_result_url"):
                    for container in _dicts(redacted_payload):
                        if key in container:
                            container[key] = "<downloaded-and-redacted>"
                write_json(raw_object_dir / "baidu_paddle_task.json", redacted_payload)

                downloads: list[Any] = []
                urls = normalized_string_values(
                    paddle_payload, ("parse_result_url", "markdown_url")
                )
                for index, url in enumerate(urls):
                    kind, downloaded = client.download_result_url(url)
                    downloaded = redact_downloaded_content(downloaded)
                    downloads.append(downloaded)
                    if kind == "json":
                        write_json(
                            raw_object_dir / f"paddle_download_{index}.json", downloaded
                        )
                    else:
                        (raw_object_dir / f"paddle_download_{index}.md").write_text(
                            str(downloaded), encoding="utf-8"
                        )
                result["processors"].append("baidu_paddleocr_vl")
                result.update(normalize_paddle(paddle_payload, downloads))
            except Exception as exc:
                if strict_specialized:
                    raise
                result["status"] = "partial"
                result["errors"].append(str(exc))
                result["quality_flags"].append("visual_analysis_failed")

    return result


def _dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _dicts(child)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--media-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--max-wait-seconds", type=float, default=300.0)
    parser.add_argument("--skip-vl", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--strict-specialized",
        action="store_true",
        help="Require table V2 and PaddleOCR-VL; disable all fallback success paths.",
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    if args.force and raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    client = BaiduOcrClient(
        require_env("BAIDU_OCR_API_KEY"),
        require_env("BAIDU_OCR_SECRET_KEY"),
        timeout=args.timeout,
        retries=args.retries,
    )
    objects = list(iter_jsonl(args.media_jsonl))
    if args.limit is not None:
        objects = objects[: args.limit]
    results_path = args.output_dir / "ocr_results.jsonl"
    previous = {
        row["object_id"]: row
        for row in iter_jsonl(results_path)
    } if results_path.exists() and not args.force else {}

    results: list[dict[str, Any]] = []
    failures = 0
    for index, obj in enumerate(objects, start=1):
        object_id = str(obj["object_id"])
        resumable_statuses = {"success"} if args.strict_specialized else {"success", "partial"}
        if object_id in previous and previous[object_id].get("status") in resumable_statuses:
            results.append(previous[object_id])
            print(f"OCR {index}/{len(objects)} resume {object_id}", flush=True)
            continue
        try:
            result = process_object(
                client,
                obj,
                raw_dir=raw_dir,
                skip_vl=args.skip_vl,
                poll_seconds=args.poll_seconds,
                max_wait_seconds=args.max_wait_seconds,
                strict_specialized=args.strict_specialized,
            )
        except Exception as exc:
            failures += 1
            result = {
                "schema_version": "paper_ocr_v1",
                "object_id": object_id,
                "paper_id": obj.get("paper_id"),
                "paper_version_id": obj.get("paper_version_id"),
                "object_type": obj.get("object_type"),
                "status": "failed",
                "error": str(exc),
                "processed_at": utc_now(),
                "quality_flags": ["ocr_failed"],
            }
            if args.fail_fast:
                results.append(result)
                write_jsonl(results_path, results)
                raise
        results.append(result)
        write_jsonl(results_path, results)
        print(
            f"OCR {index}/{len(objects)} {obj.get('object_type')} "
            f"{result.get('status')}",
            flush=True,
        )

    summary = {
        "stage": "baidu_ocr",
        "objects": len(objects),
        "success": sum(row.get("status") == "success" for row in results),
        "partial": sum(row.get("status") == "partial" for row in results),
        "failed": sum(row.get("status") == "failed" for row in results),
        "skipped": sum(row.get("status") == "skipped" for row in results),
        "skip_vl": args.skip_vl,
        "strict_specialized": args.strict_specialized,
        "output_dir": str(args.output_dir.resolve()),
    }
    write_json(args.output_dir / "ocr_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
