"""Export the live ICML RAGFlow dataset and its backend profile mapping."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from import_mode_split_format_rules import RagFlowClient, _config_value


DATASET_ID = "d6a59c2684a811f1bd3a97a1481915ff"
DOCUMENT_IDS = {
    "shared": "e8cc32bc850611f1b211d112fde53137",
    "camera_ready": "e8a6511e850611f1b211d112fde53137",
    "initial_submission": "e889e812850611f1b211d112fde53137",
}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _all_chunks(client: RagFlowClient, document_id: str) -> dict[str, Any]:
    first = client.request(
        "GET",
        f"/datasets/{DATASET_ID}/documents/{document_id}/chunks",
        params={"page": "1", "page_size": "100"},
    )
    data = first.get("data", {})
    total = int(data.get("total", 0))
    chunks = list(data.get("chunks", []))
    if len(chunks) != total:
        raise RuntimeError(
            f"Chunk export incomplete for {document_id}: got {len(chunks)}, expected {total}"
        )
    return first


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default=_config_value("RAGFLOW_BASE_URL"))
    parser.add_argument("--api-key", default=_config_value("RAGFLOW_API_KEY"))
    args = parser.parse_args()
    if not args.base_url or not args.api_key:
        raise SystemExit("RAGFLOW_BASE_URL and RAGFLOW_API_KEY are required")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    client = RagFlowClient(args.base_url, args.api_key)

    dataset = client.request("GET", f"/datasets/{DATASET_ID}")
    documents = client.request(
        "GET",
        f"/datasets/{DATASET_ID}/documents",
        params={"page": "1", "page_size": "100"},
    )
    _write_json(output_dir / "dataset.json", dataset)
    _write_json(output_dir / "documents.json", documents)

    exported_counts: dict[str, int] = {}
    for mode, document_id in DOCUMENT_IDS.items():
        payload = _all_chunks(client, document_id)
        _write_json(output_dir / f"chunks.{mode}.json", payload)
        exported_counts[mode] = len(payload["data"]["chunks"])

    profile_sql = (
        "copy (select * from format_profiles "
        "where profile_key='icml_2026' and ragflow_dataset_id='"
        + DATASET_ID
        + "') to stdout with csv header"
    )
    profile_csv = subprocess.run(
        [
            "docker",
            "exec",
            "paper-review-postgres",
            "psql",
            "-U",
            "paper_review",
            "-d",
            "paper_review",
            "-c",
            profile_sql,
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    (output_dir / "format_profile.csv").write_text(profile_csv, encoding="utf-8")

    files = sorted(path for path in output_dir.iterdir() if path.is_file())
    manifest = {
        "schema_version": "icml_format_dataset_backup_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DATASET_ID,
        "document_ids": DOCUMENT_IDS,
        "chunk_counts": exported_counts,
        "total_chunks": sum(exported_counts.values()),
        "files": {path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size} for path in files},
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
