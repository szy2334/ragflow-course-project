"""Measure ICML manifest coverage for candidate RAGFlow top-k values."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tools"))

from app.format_review.workflow import _applicable_manifest, _category_query, _match_manifest_rule_id
from import_mode_split_format_rules import RagFlowClient, _config_value

DATASET_ID = "d6a59c2684a811f1bd3a97a1481915ff"
SHARED_ID = "e8cc32bc850611f1b211d112fde53137"
MODE_IDS = {
    "camera_ready": "e8a6511e850611f1b211d112fde53137",
    "initial_submission": "e889e812850611f1b211d112fde53137",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--top-k", type=int, nargs="+", default=[24, 32, 40])
    args = parser.parse_args()
    raw_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    client = RagFlowClient(_config_value("RAGFLOW_BASE_URL") or "", _config_value("RAGFLOW_API_KEY") or "")

    jobs = []
    configurations = {}
    for mode, mode_id in MODE_IDS.items():
        manifest = _applicable_manifest(raw_manifest, mode)
        snapshot = {"venue_id": "icml", "format_version": "2026.1", "submission_mode": mode}
        categories = sorted({item["rule_category"] for item in manifest})
        configurations[mode] = (manifest, categories, [SHARED_ID, mode_id])
        for top_k in args.top_k:
            for category in categories:
                jobs.append((mode, top_k, category, _category_query(snapshot, category, manifest)))

    returned: dict[tuple[str, int, str], list[dict]] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {}
        for mode, top_k, category, query in jobs:
            _, _, documents = configurations[mode]
            future = pool.submit(
                client.request,
                "POST",
                "/retrieval",
                body={
                    "question": query,
                    "dataset_ids": [DATASET_ID],
                    "document_ids": documents,
                    "top_k": top_k,
                    "page_size": top_k,
                },
            )
            futures[future] = (mode, top_k, category)
        for future in as_completed(futures):
            key = futures[future]
            payload = future.result()
            returned[key] = list(payload.get("data", {}).get("chunks", []))

    reports = []
    for mode in MODE_IDS:
        manifest, categories, _ = configurations[mode]
        expected = {item["rule_id"] for item in manifest}
        expected_by_category = Counter(item["rule_category"] for item in manifest)
        for top_k in args.top_k:
            matched = set()
            result_counts = {}
            for category in categories:
                chunks = returned[(mode, top_k, category)]
                result_counts[category] = len(chunks)
                for chunk in chunks:
                    rule_id = _match_manifest_rule_id(str(chunk.get("content") or ""), str(chunk.get("document_id") or ""), manifest)
                    if rule_id:
                        matched.add(rule_id)
            missing = sorted(expected - matched)
            reports.append(
                {
                    "mode": mode,
                    "top_k": top_k,
                    "expected": len(expected),
                    "matched": len(matched),
                    "coverage": round(len(matched) / len(expected), 4),
                    "missing": len(missing),
                    "missing_rule_ids": missing,
                    "expected_by_category": dict(expected_by_category),
                    "returned_by_query": result_counts,
                }
            )
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
