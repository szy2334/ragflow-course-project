"""Safely replace ICML chunks in-place while preserving dataset/document IDs."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from import_mode_split_format_rules import RagFlowClient, _config_value

DATASET_ID = "d6a59c2684a811f1bd3a97a1481915ff"
DOCUMENT_IDS = {
    "shared": "e8cc32bc850611f1b211d112fde53137",
    "camera_ready": "e8a6511e850611f1b211d112fde53137",
    "initial_submission": "e889e812850611f1b211d112fde53137",
}


def _read_rows(source_dir: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for mode in DOCUMENT_IDS:
        path = source_dir / f"icml_2026_{mode}.jsonl"
        result[mode] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return result


def _backup_chunks(backup_dir: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for mode in DOCUMENT_IDS:
        payload = json.loads((backup_dir / f"chunks.{mode}.json").read_text(encoding="utf-8"))
        result[mode] = list(payload["data"]["chunks"])
    return result


def _list_chunks(client: RagFlowClient, document_id: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    page = 1
    total = 0
    while True:
        payload = client.request(
            "GET",
            f"/datasets/{DATASET_ID}/documents/{document_id}/chunks",
            params={"page": str(page), "page_size": "100"},
        )
        data = payload.get("data", {})
        batch = list(data.get("chunks", []))
        total = int(data.get("total", len(batch)))
        chunks.extend(batch)
        if len(chunks) >= total or not batch:
            break
        page += 1
    if len(chunks) != total:
        raise RuntimeError(f"Incomplete chunk listing for {document_id}: {len(chunks)}/{total}")
    return chunks


def _delete_chunks(client: RagFlowClient, document_id: str, chunk_ids: list[str]) -> None:
    if not chunk_ids:
        return
    client.request(
        "DELETE",
        f"/datasets/{DATASET_ID}/documents/{document_id}/chunks",
        body={"chunk_ids": chunk_ids},
    )


async def _update_profile(manifest: list[dict[str, Any]]) -> None:
    backend = Path(__file__).resolve().parents[1] / "backend"
    sys.path.insert(0, str(backend))
    from sqlalchemy import select
    from app.core.config import Settings
    from app.db.base import build_engine, build_session_factory
    from app.db.models import FormatProfile

    settings = Settings(_env_file=backend / ".env", _env_file_encoding="utf-8")
    engine = build_engine(settings.database_url)
    sessions = build_session_factory(engine)
    try:
        async with sessions() as session:
            profile = await session.scalar(
                select(FormatProfile).where(FormatProfile.profile_key == "icml_2026")
            )
            if profile is None:
                raise RuntimeError("ICML profile not found")
            if profile.ragflow_dataset_id != DATASET_ID:
                raise RuntimeError("ICML profile dataset ID changed since backup")
            if profile.shared_document_id != DOCUMENT_IDS["shared"]:
                raise RuntimeError("ICML shared document ID changed since backup")
            mapping = profile.mode_document_mapping_json or {}
            for mode in ("camera_ready", "initial_submission"):
                if str(mapping.get(mode)) != DOCUMENT_IDS[mode]:
                    raise RuntimeError(f"ICML {mode} document ID changed since backup")
            profile.rules_json = manifest
            await session.commit()
    finally:
        await engine.dispose()


def _restore_profile_from_backup(backup_dir: Path) -> None:
    # The replacement changes only rules_json. Extract the backed-up JSON value
    # through PostgreSQL's own CSV parser rather than trying to parse CSV here.
    raise RuntimeError(
        f"Profile restoration required; use the untouched backup at {backup_dir / 'format_profile.csv'}"
    )


def replace(client: RagFlowClient, source_dir: Path, backup_dir: Path) -> dict[str, Any]:
    rows = _read_rows(source_dir)
    backups = _backup_chunks(backup_dir)
    manifest = json.loads((source_dir / "icml_2026_rule_manifest.json").read_text(encoding="utf-8"))
    if len(manifest) != sum(len(items) for items in rows.values()):
        raise RuntimeError("Manifest and JSONL chunk counts differ")

    old_ids = {mode: {str(item["id"]) for item in items} for mode, items in backups.items()}
    desired_contents = {mode: {str(item["content"]) for item in items} for mode, items in rows.items()}
    if any(len(contents) != len(rows[mode]) for mode, contents in desired_contents.items()):
        raise RuntimeError("Duplicate cleaned chunk content detected")

    baseline = {mode: _list_chunks(client, document_id) for mode, document_id in DOCUMENT_IDS.items()}
    for mode, chunks in baseline.items():
        live_old = {str(item["id"]): str(item.get("content") or "") for item in chunks if str(item["id"]) in old_ids[mode]}
        backed_old = {str(item["id"]): str(item.get("content") or "") for item in backups[mode]}
        if live_old != backed_old:
            raise RuntimeError(f"Live {mode} chunks no longer match the verified backup")
        unexpected = [item for item in chunks if str(item["id"]) not in old_ids[mode] and str(item.get("content") or "") not in desired_contents[mode]]
        if unexpected:
            raise RuntimeError(f"Unexpected chunks exist in {mode}; refusing in-place replacement")

    created_ids: dict[str, list[str]] = {mode: [] for mode in DOCUMENT_IDS}
    try:
        for mode, document_id in DOCUMENT_IDS.items():
            current_by_content = {str(item.get("content") or ""): str(item["id"]) for item in baseline[mode]}
            for row in rows[mode]:
                content = str(row["content"])
                if content in current_by_content:
                    continue
                payload = client.request(
                    "POST",
                    f"/datasets/{DATASET_ID}/documents/{document_id}/chunks",
                    body={
                        "content": content,
                        "important_keywords": row["important_keywords"],
                        "questions": row["questions"],
                    },
                )
                created_ids[mode].append(str(payload["data"]["chunk"]["id"]))

        verified_new: dict[str, list[dict[str, Any]]] = {}
        for mode, document_id in DOCUMENT_IDS.items():
            live = _list_chunks(client, document_id)
            matches = [item for item in live if str(item.get("content") or "") in desired_contents[mode]]
            if len(matches) != len(rows[mode]) or {str(item.get("content") or "") for item in matches} != desired_contents[mode]:
                raise RuntimeError(f"Cleaned {mode} chunks failed exact read-back validation")
            verified_new[mode] = matches
    except Exception:
        for mode, document_id in DOCUMENT_IDS.items():
            _delete_chunks(client, document_id, created_ids[mode])
        raise

    try:
        asyncio.run(_update_profile(manifest))
    except Exception:
        for mode, document_id in DOCUMENT_IDS.items():
            _delete_chunks(client, document_id, created_ids[mode])
        raise

    for mode, document_id in DOCUMENT_IDS.items():
        _delete_chunks(client, document_id, sorted(old_ids[mode]))

    final: dict[str, Any] = {}
    for mode, document_id in DOCUMENT_IDS.items():
        chunks = _list_chunks(client, document_id)
        contents = {str(item.get("content") or "") for item in chunks}
        if len(chunks) != len(rows[mode]) or contents != desired_contents[mode]:
            raise RuntimeError(f"Final {mode} validation failed")
        final[mode] = {
            "document_id": document_id,
            "chunk_count": len(chunks),
            "chunk_ids": sorted(str(item["id"]) for item in chunks),
        }
    return {
        "dataset_id": DATASET_ID,
        "dataset_id_preserved": True,
        "document_ids_preserved": True,
        "old_chunk_count": sum(len(items) for items in backups.values()),
        "new_chunk_count": sum(item["chunk_count"] for item in final.values()),
        "documents": final,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--base-url", default=_config_value("RAGFLOW_BASE_URL"))
    parser.add_argument("--api-key", default=_config_value("RAGFLOW_API_KEY"))
    args = parser.parse_args()
    if not args.base_url or not args.api_key:
        raise SystemExit("RAGFLOW_BASE_URL and RAGFLOW_API_KEY are required")
    result = replace(RagFlowClient(args.base_url, args.api_key), args.source_dir, args.backup_dir)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
