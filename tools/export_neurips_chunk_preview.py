"""Export compact review records from generated NeurIPS JSONL chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = []
    for mode in ("shared", "initial_submission", "camera_ready", "preprint"):
        path = args.source_dir / f"neurips_2020_{mode}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            metadata = row["metadata"]
            records.append({
                "canonical_rule_id": metadata["canonical_rule_id"],
                "submission_mode": metadata["submission_mode"],
                "section_path": metadata["section_path"],
                "rule_category": metadata["rule_category"],
                "rule_text": metadata["source_text"],
                "important_keywords": row["important_keywords"],
                "questions": row["questions"],
                "target_document": metadata["target_document"],
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
