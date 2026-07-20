from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--base-url", default="http://localhost:9380/api/v1")
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


class RagflowApi:
    def __init__(self, base_url: str, api_key: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            }
        )

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self.session.request(
            method,
            f"{self.base_url}{path}",
            timeout=self.timeout,
            **kwargs,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError(payload.get("message") or str(payload))
        return payload.get("data")

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        data = self.request("GET", f"/datasets/{dataset_id}")
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected dataset response: {data!r}")
        return data

    def add_chunk(self, dataset_id: str, document_id: str, content: str) -> str:
        data = self.request(
            "POST",
            f"/datasets/{dataset_id}/documents/{document_id}/chunks",
            json={"content": content},
        )
        if isinstance(data, dict) and data.get("id"):
            return str(data["id"])
        if (
            isinstance(data, dict)
            and isinstance(data.get("chunk"), dict)
            and data["chunk"].get("id")
        ):
            return str(data["chunk"]["id"])
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected chunk response: {data!r}")
        raise RuntimeError(f"Unexpected chunk response: {data!r}")


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not value.get("standard_id") or not value.get("rule_text"):
                raise ValueError(f"Missing standard_id/rule_text at line {line_number}")
            records.append(value)
    return records


def render_record(record: dict[str, Any]) -> str:
    fields = [
        f"标准编号：{record['standard_id']}",
        f"论文类型：{record.get('paper_type', '')}",
        f"规则范围：{record.get('rule_scope', '')}",
        f"投稿单位代码：{record.get('venue_code', '')}",
        f"投稿单位：{record.get('venue_name', '')}",
        f"分类：{record.get('category', '')}",
        f"评价维度：{record.get('dimension', '')}",
        f"标准标题：{record.get('title', '')}",
        f"标准描述：{record['rule_text']}",
    ]
    if record.get("satisfaction_conditions"):
        fields.append("满足条件：" + "；".join(record["satisfaction_conditions"]))
    if record.get("common_defects"):
        fields.append("常见缺陷：" + "；".join(record["common_defects"]))
    if record.get("score_anchors"):
        anchors = "；".join(
            f"{key}：{value}" for key, value in record["score_anchors"].items()
        )
        fields.append(f"评分锚点：{anchors}")
    if record.get("score_min") is not None or record.get("score_max") is not None:
        fields.append(
            f"分值范围：{record.get('score_min', '')}-{record.get('score_max', '')}"
        )
    if record.get("severity"):
        fields.append(f"严重程度：{record['severity']}")
    fields.extend(
        [
            f"来源：{record.get('source_name', '')}",
            f"来源位置：{record.get('source_locator', '')}",
            f"来源网址：{record.get('source_url', '')}",
            f"标准版本：{record.get('standard_version', '')}",
            f"权威级别：{record.get('authority_level', '')}",
        ]
    )
    if record.get("validation_warning"):
        fields.append(f"使用警告：{record['validation_warning']}")
    return "\n".join(fields)


def read_state(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(path: Path, state: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("RAGFLOW_API_KEY")
    if not api_key:
        raise SystemExit("RAGFLOW_API_KEY is not set")

    state_file = args.state_file or args.source.with_suffix(".import_state.json")
    state = read_state(state_file)
    records = load_records(args.source)
    client = RagflowApi(args.base_url, api_key, args.timeout)
    dataset = client.get_dataset(args.dataset_id)
    if not dataset.get("embedding_model"):
        raise SystemExit(
            "The target RAGFlow dataset has no embedding model. Configure an "
            "embedding provider/model before importing standards."
        )

    imported = 0
    skipped = 0
    for record in records:
        standard_id = str(record["standard_id"])
        if standard_id in state:
            skipped += 1
            continue
        chunk_id = client.add_chunk(
            args.dataset_id,
            args.document_id,
            render_record(record),
        )
        state[standard_id] = chunk_id
        write_state(state_file, state)
        imported += 1
        print(f"imported {standard_id} -> {chunk_id}", flush=True)

    print(
        json.dumps(
            {
                "dataset_id": args.dataset_id,
                "document_id": args.document_id,
                "source": str(args.source),
                "records": len(records),
                "imported": imported,
                "skipped": skipped,
                "state_file": str(state_file),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
