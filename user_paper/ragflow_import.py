"""Import one user paper as manual chunks into a private RAGFlow dataset."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests

from pipeline_common import (  # noqa: E402
    config_value,
    iter_jsonl,
    load_user_paper_config,
    read_json,
    utc_now,
    write_json,
    write_jsonl,
)

load_user_paper_config()


class RagflowError(RuntimeError):
    """An unsuccessful RAGFlow API request."""


class RagflowClient:
    """Small RAGFlow API client used only by the user-paper importer."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 300.0,
        retries: int = 5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("RAGFLOW_API_KEY")
        if not self.api_key:
            raise RagflowError("RAGFLOW_API_KEY is not set")
        self.timeout = timeout
        self.retries = retries
        self._local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                }
            )
            self._local.session = session
        return session

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self._session().request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    timeout=self.timeout,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise RagflowError(
                        f"HTTP {response.status_code}: {response.text[:500]}"
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise RagflowError("RAGFlow returned a non-object JSON response")
                if payload.get("code") not in (None, 0):
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
        raise RagflowError(f"{method} {url} failed: {last_error}")

    def create_dataset(self, name: str, description: str) -> str:
        payload = self.request(
            "POST",
            "/datasets",
            json_body={
                "name": name,
                "description": description,
                "chunk_method": "manual",
            },
        )
        return str(payload["data"]["id"])

    def create_empty_document(self, dataset_id: str, name: str) -> str:
        payload = self.request(
            "POST",
            f"/datasets/{dataset_id}/documents",
            params={"type": "empty"},
            json_body={"name": name},
        )
        data = payload["data"]
        if isinstance(data, list):
            data = data[0]
        return str(data["id"])

    def delete_documents(self, dataset_id: str, document_ids: list[str]) -> None:
        if not document_ids:
            return
        self.request(
            "DELETE",
            f"/datasets/{dataset_id}/documents",
            json_body={"ids": document_ids},
        )

    def update_document(
        self, dataset_id: str, document_id: str, meta_fields: dict[str, Any]
    ) -> None:
        self.request(
            "PATCH",
            f"/datasets/{dataset_id}/documents/{document_id}",
            json_body={
                "chunk_method": "manual",
                "enabled": 1,
                "meta_fields": meta_fields,
            },
        )

    def add_chunk(self, dataset_id: str, document_id: str, content: str) -> str:
        payload = self.request(
            "POST",
            f"/datasets/{dataset_id}/documents/{document_id}/chunks",
            json_body={
                "content": content,
                "important_keywords": [],
                "questions": [],
                "tag_kwd": [],
            },
        )
        return str(payload["data"]["chunk"]["id"])


SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    paper_version_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    ragflow_document_id TEXT NOT NULL,
    document_name TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    source_chunk_id TEXT PRIMARY KEY,
    ragflow_chunk_id TEXT,
    content_sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    updated_at TEXT NOT NULL
);
"""


def open_state(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def metadata_get(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return None if row is None else str(row[0])


def metadata_set(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        "INSERT INTO metadata(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    connection.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument(
        "--base-url", default=config_value("RAGFLOW_BASE_URL", "http://localhost:9380/api/v1")
    )
    parser.add_argument("--dataset-id")
    parser.add_argument(
        "--dataset-name",
        default=config_value("USER_PAPER_DATASET_NAME", "user_papers_private_v1"),
    )
    parser.add_argument("--user-id", default=config_value("USER_PAPER_USER_ID", "local-user"))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--replace-document",
        action="store_true",
        help="Delete the previously imported document for this paper before importing fresh chunks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = read_json(args.chunks_dir / "document_manifest.json")
    chunks = list(iter_jsonl(args.chunks_dir / "ragflow_chunks.jsonl"))
    quality = read_json(args.chunks_dir / "quality_report.json")
    if quality.get("status") == "FAILED":
        raise SystemExit("Chunk quality status is FAILED; refusing RAGFlow import")
    if not chunks:
        raise SystemExit("No indexable chunks found")

    args.state_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        summary = {
            "dry_run": True,
            "dataset_name": args.dataset_name,
            "paper_id": manifest["paper_id"],
            "paper_version_id": manifest["paper_version_id"],
            "document_name": manifest["document_name"],
            "chunks": len(chunks),
            "quality_status": quality.get("status"),
        }
        write_json(args.state_dir / "import_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 0

    state_db = args.state_dir / "import_state.sqlite"
    connection = open_state(state_db)
    client = RagflowClient(
        args.base_url,
        timeout=args.timeout,
        retries=args.retries,
    )

    stored_dataset_id = metadata_get(connection, "dataset_id")
    if args.dataset_id and stored_dataset_id and args.dataset_id != stored_dataset_id:
        raise RuntimeError(
            f"State belongs to dataset {stored_dataset_id}, not {args.dataset_id}"
        )
    dataset_id = args.dataset_id or stored_dataset_id
    if dataset_id is None:
        dataset_id = client.create_dataset(
            args.dataset_name,
            "Private user-uploaded scientific papers imported as structured manual chunks",
        )
        metadata_set(connection, "dataset_id", dataset_id)
        metadata_set(connection, "dataset_name", args.dataset_name)

    paper_version_id = str(manifest["paper_version_id"])
    existing_document = connection.execute(
        "SELECT ragflow_document_id,status FROM documents WHERE paper_version_id=?",
        (paper_version_id,),
    ).fetchone()
    if existing_document is not None and args.replace_document:
        client.delete_documents(dataset_id, [str(existing_document[0])])
        # State directories are paper-specific, so clearing these rows cannot
        # affect another paper's document in the same RAGFlow dataset.
        connection.execute("DELETE FROM chunks")
        connection.execute(
            "DELETE FROM documents WHERE paper_version_id=?", (paper_version_id,)
        )
        connection.commit()
        existing_document = None
    if existing_document is None:
        ragflow_document_id = client.create_empty_document(
            dataset_id, str(manifest["document_name"])
        )
        connection.execute(
            "INSERT INTO documents VALUES(?,?,?,?,?,?)",
            (
                paper_version_id,
                dataset_id,
                ragflow_document_id,
                manifest["document_name"],
                "created",
                utc_now(),
            ),
        )
        connection.commit()
    else:
        ragflow_document_id = str(existing_document[0])

    meta_fields = dict(manifest.get("meta_fields") or {})
    meta_fields.update(
        {
            "user_id": args.user_id,
            "paper_id": manifest["paper_id"],
            "paper_version_id": paper_version_id,
            "ingest_source": "user_pdf",
            "quality_status": quality.get("status"),
        }
    )
    client.update_document(dataset_id, ragflow_document_id, meta_fields)
    connection.execute(
        "UPDATE documents SET status='ready' WHERE paper_version_id=?",
        (paper_version_id,),
    )
    connection.commit()

    imported = 0
    resumed = 0
    failures = 0
    mapping: list[dict[str, Any]] = []
    for index, row in enumerate(chunks, start=1):
        source_chunk_id = str(row["document_id"])
        metadata = row.get("metadata") or {}
        content_sha256 = str(metadata.get("content_sha256") or "")
        if not content_sha256:
            import hashlib

            content_sha256 = hashlib.sha256(row["content"].encode("utf-8")).hexdigest()
        existing = connection.execute(
            "SELECT ragflow_chunk_id,status,content_sha256 FROM chunks "
            "WHERE source_chunk_id=?",
            (source_chunk_id,),
        ).fetchone()
        if existing and existing[1] == "ok" and existing[2] == content_sha256:
            ragflow_chunk_id = str(existing[0])
            resumed += 1
        else:
            try:
                ragflow_chunk_id = client.add_chunk(
                    dataset_id, ragflow_document_id, str(row["content"])
                )
                connection.execute(
                    "INSERT INTO chunks VALUES(?,?,?,'ok',NULL,?) "
                    "ON CONFLICT(source_chunk_id) DO UPDATE SET "
                    "ragflow_chunk_id=excluded.ragflow_chunk_id,"
                    "content_sha256=excluded.content_sha256,status='ok',error=NULL,"
                    "updated_at=excluded.updated_at",
                    (source_chunk_id, ragflow_chunk_id, content_sha256, utc_now()),
                )
                connection.commit()
                imported += 1
            except Exception as exc:
                failures += 1
                connection.execute(
                    "INSERT INTO chunks VALUES(?,NULL,?,'error',?,?) "
                    "ON CONFLICT(source_chunk_id) DO UPDATE SET "
                    "status='error',error=excluded.error,updated_at=excluded.updated_at",
                    (source_chunk_id, content_sha256, repr(exc), utc_now()),
                )
                connection.commit()
                print(f"RAGFlow chunk {index}/{len(chunks)} failed: {exc}", file=sys.stderr)
                continue
        # This file is also the stable hand-off from the ingestion pipeline to
        # the asynchronous AI retrieval adapter.  Keep enough provenance here
        # to rebuild an EvidenceItem after RAGFlow returns only a chunk ID and
        # content.  No secret or full chunk text is written to the mapping.
        mapping.append(
            {
                "schema_version": "user_paper_ragflow_mapping_v2",
                "user_id": args.user_id,
                "source_chunk_id": source_chunk_id,
                "ragflow_chunk_id": ragflow_chunk_id,
                "paper_id": manifest["paper_id"],
                "paper_version_id": paper_version_id,
                "dataset_id": dataset_id,
                "document_id": ragflow_document_id,
                "source_ref": metadata.get("source_ref"),
                "content_type": metadata.get("content_type"),
                "content_role": metadata.get("content_role"),
                "section": metadata.get("section"),
                "section_path": metadata.get("section_path"),
                "page_start": metadata.get("page_start"),
                "page_end": metadata.get("page_end"),
                "object_id": metadata.get("object_id"),
                "parent_chunk_id": metadata.get("parent_chunk_id"),
                "quality_flags": metadata.get("quality_flags") or [],
            }
        )
        if index % 20 == 0 or index == len(chunks):
            print(
                f"RAGFlow chunks {index}/{len(chunks)} imported={imported} "
                f"resumed={resumed} failed={failures}",
                flush=True,
            )

    write_jsonl(args.state_dir / "chunk_mapping.jsonl", mapping)
    summary = {
        "mapping_schema_version": "user_paper_ragflow_mapping_v2",
        "dataset_id": dataset_id,
        "dataset_name": args.dataset_name,
        "document_id": ragflow_document_id,
        "user_id": args.user_id,
        "paper_id": manifest["paper_id"],
        "paper_version_id": paper_version_id,
        "expected_chunks": len(chunks),
        "mapped_chunks": len(mapping),
        "imported_this_run": imported,
        "resumed": resumed,
        "failures": failures,
        "quality_status": quality.get("status"),
        "state_db": str(state_db.resolve()),
        "completed_at": utc_now(),
    }
    write_json(args.state_dir / "import_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if failures == 0 and len(mapping) == len(chunks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
