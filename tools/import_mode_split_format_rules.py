"""Replace one legacy RAGFlow rule document with mode-scoped manual documents."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


class RagFlowError(RuntimeError):
    pass


class RagFlowClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RagFlowError(f"{method} {url} failed") from exc
        if payload.get("code") != 0:
            raise RagFlowError(f"{method} {path} failed: {payload.get('message')}")
        return payload

    def list_documents(self, dataset_id: str) -> list[dict[str, Any]]:
        payload = self.request(
            "GET", f"/datasets/{dataset_id}/documents", params={"page": "1", "page_size": "100"}
        )
        return list(payload.get("data", {}).get("docs", []))

    def create_document(self, dataset_id: str, name: str, meta_fields: dict[str, str]) -> str:
        payload = self.request(
            "POST",
            f"/datasets/{dataset_id}/documents",
            params={"type": "empty"},
            body={"name": name},
        )
        data = payload["data"]
        record = data[0] if isinstance(data, list) else data
        document_id = str(record["id"])
        self.request(
            "PATCH",
            f"/datasets/{dataset_id}/documents/{document_id}",
            body={"chunk_method": "manual", "enabled": 1, "meta_fields": meta_fields},
        )
        return document_id

    def add_chunk(self, dataset_id: str, document_id: str, row: dict[str, Any]) -> str:
        metadata = row["metadata"]
        keywords = _keywords(str(row["content"]), metadata)
        payload = self.request(
            "POST",
            f"/datasets/{dataset_id}/documents/{document_id}/chunks",
            body={"content": row["content"], "important_keywords": keywords, "questions": []},
        )
        return str(payload["data"]["chunk"]["id"])

    def list_chunks(self, dataset_id: str, document_id: str) -> list[dict[str, Any]]:
        payload = self.request(
            "GET",
            f"/datasets/{dataset_id}/documents/{document_id}/chunks",
            params={"page": "1", "page_size": "100"},
        )
        return list(payload.get("data", {}).get("chunks", []))

    def delete_documents(self, dataset_id: str, document_ids: list[str]) -> None:
        if document_ids:
            self.request("DELETE", f"/datasets/{dataset_id}/documents", body={"ids": document_ids})


def _config_value(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    env_path = Path("backend/.env")
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _keywords(content: str, metadata: dict[str, Any]) -> list[str]:
    values = [
        str(metadata["submission_mode"]),
        str(metadata["rule_category"]),
        str(metadata["venue_id"]),
        str(metadata["format_version"]),
    ]
    match = re.search(r"检索关键词：([^\n]+)", content)
    if match:
        values.extend(part.strip() for part in re.split(r"[；;、,]", match.group(1)))
    return list(dict.fromkeys(value for value in values if value))


def _read_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            groups[str(row["metadata"]["target_document"])].append(row)
    if not groups:
        raise RagFlowError(f"{path} contains no rule chunks")
    return dict(groups)


def _document_metadata(rows: list[dict[str, Any]]) -> dict[str, str]:
    item = rows[0]["metadata"]
    return {
        "venue_id": str(item["venue_id"]),
        "format_version": str(item["format_version"]),
        "submission_mode": str(item["submission_mode"]),
        "rule_scope": "shared" if item["submission_mode"] == "shared" else "mode_specific",
    }


def _document_name(target_document: str) -> str:
    return f"{target_document}.jsonl"


def import_rules(
    client: RagFlowClient,
    *,
    dataset_id: str,
    legacy_document_name: str,
    source: Path,
    manifest_path: Path,
    dry_run: bool,
    max_chunks: int | None,
) -> dict[str, Any]:
    groups = _read_rows(source)
    existing = client.list_documents(dataset_id)
    legacy = [doc for doc in existing if doc.get("name") == legacy_document_name]
    if len(legacy) != 1:
        raise RagFlowError(f"dataset {dataset_id} must contain exactly one legacy document")

    existing_by_name = {str(doc["name"]): doc for doc in existing}
    allowed_names = {legacy_document_name, *(_document_name(target) for target in groups)}
    unexpected = sorted(set(existing_by_name) - allowed_names)
    if unexpected:
        raise RagFlowError(f"dataset {dataset_id} has unexpected documents: {unexpected}")

    summary = {
        "dataset_id": dataset_id,
        "legacy_document": {"id": legacy[0]["id"], "name": legacy_document_name},
        "planned_documents": {
            target: {"name": _document_name(target), "chunks": len(rows)}
            for target, rows in groups.items()
        },
    }
    if dry_run:
        return {**summary, "dry_run": True}

    created: dict[str, dict[str, Any]] = {}
    newly_created_document_ids: list[str] = []
    try:
        for target, rows in groups.items():
            name = _document_name(target)
            current = existing_by_name.get(name)
            if current is not None:
                document_id = str(current["id"])
            else:
                document_id = client.create_document(
                    dataset_id, name, _document_metadata(rows)
                )
                newly_created_document_ids.append(document_id)
            existing_contents = {
                str(chunk.get("content", ""))
                for chunk in client.list_chunks(dataset_id, document_id)
            }
            created[target] = {
                "document_id": document_id,
                "document_name": name,
                "chunk_count": len(rows),
                "chunk_ids": [],
                "missing_rows": [row for row in rows if row["content"] not in existing_contents],
                "submission_mode": rows[0]["metadata"]["submission_mode"],
            }
        jobs = [
            (target, created[target]["document_id"], row)
            for target in groups
            for row in created[target]["missing_rows"]
        ]
        if max_chunks is not None:
            jobs = jobs[:max_chunks]
        for target, document_id, row in jobs:
            created[target]["chunk_ids"].append(client.add_chunk(dataset_id, document_id, row))
    except Exception:
        client.delete_documents(dataset_id, newly_created_document_ids)
        raise

    verified = {doc["id"]: doc for doc in client.list_documents(dataset_id)}
    remaining: dict[str, int] = {}
    for item in created.values():
        document = verified.get(item["document_id"])
        actual = int(document.get("chunk_count", 0)) if document is not None else 0
        if actual > item["chunk_count"]:
            raise RagFlowError("RAGFlow document contains more chunks than the approved source")
        if actual < item["chunk_count"]:
            remaining[item["document_name"]] = item["chunk_count"] - actual

    if remaining:
        return {
            **summary,
            "dry_run": False,
            "complete": False,
            "imported_this_run": len(jobs),
            "remaining_chunks": remaining,
        }

    client.delete_documents(dataset_id, [str(legacy[0]["id"])])
    result = {
        **summary,
        "dry_run": False,
        "complete": True,
        "created_documents": {
            target: {key: value for key, value in item.items() if key != "missing_rows"}
            for target, item in created.items()
        },
    }
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--legacy-document-name", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-url", default=_config_value("RAGFLOW_BASE_URL"))
    parser.add_argument("--api-key", default=_config_value("RAGFLOW_API_KEY"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-chunks", type=int)
    args = parser.parse_args()
    if not args.base_url or not args.api_key:
        raise SystemExit("RAGFLOW_BASE_URL and RAGFLOW_API_KEY are required")
    result = import_rules(
        RagFlowClient(args.base_url, args.api_key),
        dataset_id=args.dataset_id,
        legacy_document_name=args.legacy_document_name,
        source=args.source,
        manifest_path=args.manifest,
        dry_run=args.dry_run,
        max_chunks=args.max_chunks,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
