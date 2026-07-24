"""Replace NeurIPS 2020 rule chunks while preserving existing IDs and mappings."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from import_mode_split_format_rules import RagFlowClient, _config_value

DATASET_ID = "e3c8be1a84a811f1bd3a97a1481915ff"
DOCUMENT_IDS = {
    "shared": "6bd5126e850711f1b211d112fde53137",
    "initial_submission": "6becebd2850711f1b211d112fde53137",
    "camera_ready": "6c0d37d4850711f1b211d112fde53137",
    "preprint": "6c294fe6850711f1b211d112fde53137",
}
EXPECTED_OLD_COUNTS = {"shared": 14, "initial_submission": 4, "camera_ready": 4, "preprint": 1}


def _read_rows(source_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        mode: [json.loads(line) for line in (source_dir / f"neurips_2020_{mode}.jsonl").read_text(encoding="utf-8").splitlines() if line]
        for mode in DOCUMENT_IDS
    }


def _list_chunks(client: RagFlowClient, document_id: str) -> list[dict[str, Any]]:
    payload = client.request("GET", f"/datasets/{DATASET_ID}/documents/{document_id}/chunks", params={"page": "1", "page_size": "100"})
    data = payload.get("data", {})
    chunks = list(data.get("chunks", []))
    if len(chunks) != int(data.get("total", len(chunks))):
        raise RuntimeError(f"Incomplete chunk listing for {document_id}")
    return chunks


def _delete_chunks(client: RagFlowClient, document_id: str, chunk_ids: list[str]) -> None:
    if chunk_ids:
        client.request("DELETE", f"/datasets/{DATASET_ID}/documents/{document_id}/chunks", body={"chunk_ids": chunk_ids})


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
            profile = await session.scalar(select(FormatProfile).where(FormatProfile.profile_key == "neurips_2020"))
            if profile is None or profile.ragflow_dataset_id != DATASET_ID:
                raise RuntimeError("NeurIPS profile dataset mapping changed")
            if profile.shared_document_id != DOCUMENT_IDS["shared"]:
                raise RuntimeError("NeurIPS shared document mapping changed")
            mapping = profile.mode_document_mapping_json or {}
            for mode in ("initial_submission", "camera_ready", "preprint"):
                if str(mapping.get(mode)) != DOCUMENT_IDS[mode]:
                    raise RuntimeError(f"NeurIPS {mode} document mapping changed")
            profile.rules_json = manifest
            await session.commit()
    finally:
        await engine.dispose()


def replace(client: RagFlowClient, source_dir: Path) -> dict[str, Any]:
    rows = _read_rows(source_dir)
    manifest = json.loads((source_dir / "neurips_2020_rule_manifest.json").read_text(encoding="utf-8"))
    if len(manifest) != sum(len(items) for items in rows.values()) or len(rows["shared"]) != 35:
        raise RuntimeError("NeurIPS source count or manifest mismatch")
    baseline = {mode: _list_chunks(client, doc) for mode, doc in DOCUMENT_IDS.items()}
    for mode, chunks in baseline.items():
        if len(chunks) != EXPECTED_OLD_COUNTS[mode]:
            raise RuntimeError(f"Unexpected live {mode} count: {len(chunks)}")
    desired = {mode: {str(row["content"]): row for row in items} for mode, items in rows.items()}
    created: list[tuple[str, str]] = []
    try:
        for row in rows["shared"]:
            payload = client.request("POST", f"/datasets/{DATASET_ID}/documents/{DOCUMENT_IDS['shared']}/chunks", body={"content": row["content"], "important_keywords": row["important_keywords"], "questions": row["questions"]})
            created.append((DOCUMENT_IDS["shared"], str(payload["data"]["chunk"]["id"])))
        live_shared = _list_chunks(client, DOCUMENT_IDS["shared"])
        live_contents = {str(item.get("content") or "") for item in live_shared}
        if not desired["shared"].keys() <= live_contents:
            raise RuntimeError("New shared chunks failed read-back validation")
        asyncio.run(_update_profile(manifest))
        for mode, document_id in DOCUMENT_IDS.items():
            _delete_chunks(client, document_id, [str(item["id"]) for item in baseline[mode]])
        final = {mode: _list_chunks(client, doc) for mode, doc in DOCUMENT_IDS.items()}
        if len(final["shared"]) != 35 or any(final[mode] for mode in ("initial_submission", "camera_ready", "preprint")):
            raise RuntimeError("Final NeurIPS document counts failed")
        return {
            "dataset_id": DATASET_ID,
            "document_ids_preserved": True,
            "old_chunk_count": sum(len(items) for items in baseline.values()),
            "new_chunk_count": len(final["shared"]),
            "documents": {mode: {"document_id": doc, "chunk_count": len(final[mode])} for mode, doc in DOCUMENT_IDS.items()},
        }
    except Exception:
        _delete_chunks(client, DOCUMENT_IDS["shared"], [chunk_id for doc, chunk_id in created if doc == DOCUMENT_IDS["shared"]])
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--base-url", default=_config_value("RAGFLOW_BASE_URL"))
    parser.add_argument("--api-key", default=_config_value("RAGFLOW_API_KEY"))
    args = parser.parse_args()
    if not args.base_url or not args.api_key:
        raise SystemExit("RAGFLOW_BASE_URL and RAGFLOW_API_KEY are required")
    result = replace(RagFlowClient(args.base_url, args.api_key), args.source_dir)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
