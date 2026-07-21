"""Run golden QA questions against the RAGFlow retrieval API.

The generated results file is the input to ``qa_baseline.py``.  The executor
only calls RAGFlow's retrieval endpoint and measures whether it returns the
expected document chunks for each question.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from pipeline_common import config_bool, config_value, load_user_paper_config, utc_now, write_json


load_user_paper_config()


class RagflowError(RuntimeError):
    """An unsuccessful RAGFlow API request."""


class RagflowRetrievalClient:
    def __init__(self, base_url: str, *, timeout: float, retries: int) -> None:
        api_key = os.environ.get("RAGFLOW_API_KEY")
        if not api_key:
            raise RagflowError("RAGFLOW_API_KEY is not set")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            }
        )

    def retrieve(
        self,
        question: str,
        *,
        dataset_id: str,
        document_id: str,
        page_size: int,
        similarity_threshold: float,
        vector_similarity_weight: float,
        cross_languages: list[str],
    ) -> dict[str, Any]:
        url = f"{self.base_url}/retrieval"
        body = {
            "question": question,
            "dataset_ids": [dataset_id],
            "document_ids": [document_id],
            "page": 1,
            "page_size": page_size,
            "similarity_threshold": similarity_threshold,
            "vector_similarity_weight": vector_similarity_weight,
        }
        if cross_languages:
            body["cross_languages"] = cross_languages
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.session.post(url, json=body, timeout=self.timeout)
                if response.status_code == 429 or response.status_code >= 500:
                    raise RagflowError(f"HTTP {response.status_code}: {response.text[:500]}")
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise RagflowError("RAGFlow returned a non-object JSON response")
                if payload.get("code") not in (None, 0, "0"):
                    raise RagflowError(
                        f"RAGFlow code {payload.get('code')}: "
                        f"{payload.get('message', payload)}"
                    )
                return payload
            except (requests.RequestException, ValueError, RagflowError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(2**attempt, 20))
        raise RagflowError(f"POST {url} failed: {last_error}")


def configured_path(name: str) -> Path | None:
    value = config_value(name)
    return Path(value).expanduser() if value else None


def parse_csv_values(value: str | None) -> list[str]:
    """Parse a comma-separated config value into a stable non-empty list."""
    values = (part.strip() for part in (value or "").split(","))
    return list(dict.fromkeys(part for part in values if part))


def retrieval_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return all request settings that change retrieval results."""
    return {
        "page_size": args.page_size,
        "similarity_threshold": args.similarity_threshold,
        "vector_similarity_weight": args.vector_similarity_weight,
        "cross_languages": args.cross_languages,
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def references_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data") or {}
    if not isinstance(data, dict):
        raise RagflowError("RAGFlow retrieval response has no object data field")
    raw_chunks = data.get("chunks") or []
    if not raw_chunks:
        reference = data.get("reference") or {}
        raw_chunks = reference.get("chunks") if isinstance(reference, dict) else []
    if not isinstance(raw_chunks, list):
        raw_chunks = []
    references: list[dict[str, Any]] = []
    for chunk in raw_chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_id = chunk.get("id") or chunk.get("chunk_id")
        if not chunk_id:
            continue
        references.append(
            {
                "id": str(chunk_id),
                "content": str(chunk.get("content") or ""),
                "document_name": str(
                    chunk.get("document_name")
                    or chunk.get("document_keyword")
                    or chunk.get("docnm_kwd")
                    or ""
                ),
            }
        )
    return references


def resolve_target(
    mapping_path: Path, dataset_id: str | None, document_id: str | None
) -> tuple[str, str]:
    dataset_ids: set[str] = set()
    document_ids: set[str] = set()
    with mapping_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {mapping_path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {mapping_path}:{line_number}")
            if row.get("dataset_id"):
                dataset_ids.add(str(row["dataset_id"]))
            if row.get("document_id"):
                document_ids.add(str(row["document_id"]))
    resolved_dataset_id = dataset_id or (next(iter(dataset_ids)) if len(dataset_ids) == 1 else None)
    resolved_document_id = document_id or (next(iter(document_ids)) if len(document_ids) == 1 else None)
    if not resolved_dataset_id:
        raise ValueError(
            "Could not determine one dataset ID from the chunk mapping; pass --dataset-id."
        )
    if not resolved_document_id:
        raise ValueError(
            "Could not determine one document ID from the chunk mapping; pass --document-id."
        )
    return resolved_dataset_id, resolved_document_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden",
        type=Path,
        default=configured_path("USER_PAPER_QA_GOLDEN"),
        help="Golden QA cases (default: USER_PAPER_QA_GOLDEN).",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=configured_path("USER_PAPER_QA_RESULTS"),
        help="Results JSON output (default: USER_PAPER_QA_RESULTS).",
    )
    parser.add_argument(
        "--chunk-mapping",
        type=Path,
        default=configured_path("USER_PAPER_QA_CHUNK_MAPPING"),
        help="RAGFlow chunk mapping (default: USER_PAPER_QA_CHUNK_MAPPING).",
    )
    parser.add_argument("--dataset-id", default=config_value("USER_PAPER_QA_DATASET_ID"))
    parser.add_argument("--document-id", default=config_value("USER_PAPER_QA_DOCUMENT_ID"))
    parser.add_argument(
        "--base-url",
        default=config_value("RAGFLOW_BASE_URL", "http://localhost:9380/api/v1"),
    )
    parser.add_argument("--timeout", type=float, default=float(config_value("USER_PAPER_QA_TIMEOUT", "300")))
    parser.add_argument("--retries", type=int, default=int(config_value("USER_PAPER_QA_RETRIES", "2")))
    parser.add_argument(
        "--page-size",
        type=int,
        default=int(config_value("USER_PAPER_QA_PAGE_SIZE", "8")),
        help="Maximum retrieved chunks per question (default: USER_PAPER_QA_PAGE_SIZE).",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=float(config_value("USER_PAPER_QA_SIMILARITY_THRESHOLD", "0")),
    )
    parser.add_argument(
        "--vector-similarity-weight",
        type=float,
        default=float(config_value("USER_PAPER_QA_VECTOR_SIMILARITY_WEIGHT", "0.3")),
    )
    parser.add_argument(
        "--cross-languages",
        default=config_value("USER_PAPER_QA_CROSS_LANGUAGES", ""),
        metavar="LANGUAGE[,LANGUAGE...]",
        help=(
            "Comma-separated target languages for RAGFlow cross-language retrieval "
            "(default: USER_PAPER_QA_CROSS_LANGUAGES; example: English)."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=config_bool("USER_PAPER_QA_RESUME", True),
        help="Reuse successful matching questions in an existing retrieval-results file.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    missing = [
        f"{option} (set {setting} in .env)"
        for option, setting, value in (
            ("--golden", "USER_PAPER_QA_GOLDEN", args.golden),
            ("--results", "USER_PAPER_QA_RESULTS", args.results),
            ("--chunk-mapping", "USER_PAPER_QA_CHUNK_MAPPING", args.chunk_mapping),
        )
        if not value
    ]
    if missing:
        parser.error("Missing configuration: " + ", ".join(missing))
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.retries < 0:
        parser.error("--retries cannot be negative")
    if args.page_size <= 0:
        parser.error("--page-size must be positive")
    if not 0 <= args.similarity_threshold <= 1:
        parser.error("--similarity-threshold must be between 0 and 1")
    if not 0 <= args.vector_similarity_weight <= 1:
        parser.error("--vector-similarity-weight must be between 0 and 1")
    args.cross_languages = parse_csv_values(args.cross_languages)
    return args


def main() -> int:
    args = parse_args()
    request_config = retrieval_config(args)
    golden = read_json(args.golden)
    cases = golden.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise SystemExit(f"Golden file has no cases: {args.golden}")
    invalid = [
        index + 1
        for index, case in enumerate(cases)
        if not isinstance(case, dict) or not case.get("question_zh")
    ]
    if invalid:
        raise SystemExit(f"Golden cases missing question_zh at positions: {invalid}")
    dataset_id, document_id = resolve_target(
        args.chunk_mapping, args.dataset_id, args.document_id
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "evaluation_mode": "retrieval",
                    "dataset_id": dataset_id,
                    "document_id": document_id,
                    "golden": str(args.golden),
                    "results": str(args.results),
                    "questions": len(cases),
                    "retrieval_config": request_config,
                },
                ensure_ascii=False,
            )
        )
        return 0

    existing_by_id: dict[str, dict[str, Any]] = {}
    if args.resume and args.results.exists():
        existing = read_json(args.results)
        if (
            existing.get("evaluation_mode") == "retrieval"
            and existing.get("retrieval_config") == request_config
        ):
            for row in existing.get("results") or []:
                if isinstance(row, dict) and row.get("id") and not row.get("error"):
                    existing_by_id[str(row["id"])] = row
        else:
            print(
                "Existing retrieval results use different settings; skipping resume.",
                flush=True,
            )

    client = RagflowRetrievalClient(args.base_url, timeout=args.timeout, retries=args.retries)
    results: list[dict[str, Any]] = []
    executed = 0
    resumed = 0
    failures = 0
    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("id") or f"case_{index}")
        question = str(case["question_zh"])
        existing = existing_by_id.get(case_id)
        if existing and str(existing.get("question") or "") == question:
            results.append(existing)
            resumed += 1
            continue
        try:
            payload = client.retrieve(
                question,
                dataset_id=dataset_id,
                document_id=document_id,
                page_size=args.page_size,
                similarity_threshold=args.similarity_threshold,
                vector_similarity_weight=args.vector_similarity_weight,
                cross_languages=args.cross_languages,
            )
            references = references_from_payload(payload)
            results.append(
                {
                    "id": case_id,
                    "kind": case.get("result_kind"),
                    "question": question,
                    "references": references,
                }
            )
            executed += 1
            print(f"Retrieval {index}/{len(cases)} completed; chunks={len(references)}", flush=True)
        except Exception as exc:
            failures += 1
            results.append(
                {
                    "id": case_id,
                    "kind": case.get("result_kind"),
                    "question": question,
                    "references": [],
                    "error": str(exc),
                }
            )
            print(f"Retrieval {index}/{len(cases)} failed: {exc}", flush=True)
            if args.fail_fast:
                break

    output = {
        "schema_version": "user_paper_qa_retrieval_results_v1",
        "generated_at": utc_now(),
        "evaluation_mode": "retrieval",
        "dataset_id": dataset_id,
        "document_id": document_id,
        "golden_file": str(args.golden),
        "retrieval_config": request_config,
        "results": results,
    }
    write_json(args.results, output)
    summary = {
        "questions": len(cases),
        "executed": executed,
        "resumed": resumed,
        "failures": failures,
        "retrieval_config": request_config,
        "results_file": str(args.results),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
