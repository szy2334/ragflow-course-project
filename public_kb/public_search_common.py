"""Shared standalone retrieval runner for one public RAGFlow dataset."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests


SIMILARITY_THRESHOLD = 0.2
DEFAULT_TOP_K = 1024
DEFAULT_SIZE = 12
DEFAULT_ANSWER_LIMIT = 4
DEFAULT_VECTOR_WEIGHT = 0.3
DEFAULT_TIMEOUT = 60.0
MANIFEST_PATH = Path(__file__).resolve().with_name("manifest.json")


@dataclass(frozen=True)
class DatasetProfile:
    key: str
    dataset_name: str
    display_name: str
    indexed_document_name: str
    questions: tuple[str, ...]


def configure_console() -> None:
    if os.name != "nt":
        return
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def normalize_api_key(value: str | None) -> str:
    key = (value or "").strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in {"'", '"'}:
        key = key[1:-1].strip()
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key


def normalize_base_url(value: str) -> str:
    base_url = value.strip().rstrip("/")
    if not base_url:
        base_url = "http://localhost:9380/api/v1"
    if not base_url.endswith("/api/v1"):
        base_url += "/api/v1"
    return base_url


def load_manifest() -> dict[str, Any]:
    try:
        value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取公共库清单 {MANIFEST_PATH}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"公共库清单格式错误：{MANIFEST_PATH}")
    return value


def find_manifest_dataset(
    manifest: Mapping[str, Any], dataset_name: str
) -> dict[str, Any]:
    for section in ("active_datasets", "content_datasets", "venue_format_datasets"):
        entries = manifest.get(section, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and entry.get("dataset_name") == dataset_name:
                return entry
    raise RuntimeError(f"manifest.json 中找不到公共库 {dataset_name}")


def metadata(chunk: Mapping[str, Any]) -> Mapping[str, Any]:
    value = chunk.get("metadata")
    return value if isinstance(value, Mapping) else {}


def field(chunk: Mapping[str, Any], name: str) -> Any:
    value = metadata(chunk).get(name)
    return chunk.get(name) if value is None else value


def score_of(chunk: Mapping[str, Any]) -> float | None:
    for name in ("rerank_score", "similarity", "score"):
        value = field(chunk, name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def content_of(chunk: Mapping[str, Any]) -> str:
    for value in (
        chunk.get("content_with_weight"),
        chunk.get("content"),
        chunk.get("text"),
        field(chunk, "rule_text"),
        chunk.get("content_ltks"),
    ):
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return ""


def labeled_value(chunk: Mapping[str, Any], label: str) -> str | None:
    raw = chunk.get("content_with_weight")
    if not isinstance(raw, str):
        return None
    match = re.search(rf"(?:^|\n){re.escape(label)}[：:]\s*([^\n]+)", raw)
    return match.group(1).strip() if match else None


def extract_chunks(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = data.get("chunks") or data.get("chunk") or []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def filter_document_chunks(
    chunks: Sequence[Mapping[str, Any]],
    document_id: str | None,
    document_name: str | None = None,
) -> list[dict[str, Any]]:
    if not document_id and not document_name:
        return [dict(chunk) for chunk in chunks]
    scoped: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_document_id = str(
            chunk.get("document_id") or chunk.get("doc_id") or ""
        )
        chunk_document_name = str(
            chunk.get("document_name")
            or chunk.get("doc_name")
            or chunk.get("docnm_kwd")
            or ""
        )
        if (document_id and chunk_document_id == document_id) or (
            document_name and chunk_document_name == document_name
        ):
            scoped.append(dict(chunk))
    return scoped


def rank_chunks(
    chunks: Sequence[Mapping[str, Any]],
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for chunk in chunks:
        score = score_of(chunk)
        if score is None or score < threshold or not content_of(chunk):
            continue
        ranked.append(dict(chunk))
    ranked.sort(key=lambda item: score_of(item) or 0.0, reverse=True)
    return ranked


def compact_text(value: str, limit: int = 800) -> str:
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


def answer_excerpt(chunk: Mapping[str, Any]) -> str:
    title = labeled_value(chunk, "标准标题")
    description = labeled_value(chunk, "标准描述")
    details = []
    for label in ("满足条件", "评分锚点", "分值范围"):
        value = labeled_value(chunk, label)
        if value:
            details.append(f"{label}：{value}")
    if title and description:
        main = f"{title}：{description}"
    else:
        main = description or title or content_of(chunk)
    if details:
        main += " " + " ".join(details)
    return compact_text(main)


def evidence_view(chunk: Mapping[str, Any], rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "score": score_of(chunk),
        "standard_id": field(chunk, "standard_id")
        or labeled_value(chunk, "标准编号"),
        "standard_version": field(chunk, "standard_version")
        or labeled_value(chunk, "标准版本"),
        "document_id": chunk.get("document_id") or chunk.get("doc_id"),
        "document_name": chunk.get("document_name")
        or chunk.get("doc_name")
        or chunk.get("docnm_kwd")
        or "未知文档",
        "chunk_id": chunk.get("chunk_id") or chunk.get("id"),
        "content": content_of(chunk),
    }


def build_answer(
    display_name: str,
    ranked_chunks: Sequence[Mapping[str, Any]],
    answer_limit: int = DEFAULT_ANSWER_LIMIT,
) -> tuple[str, list[dict[str, Any]]]:
    selected = list(ranked_chunks[:answer_limit])
    evidences = [evidence_view(chunk, index) for index, chunk in enumerate(selected, 1)]
    if not evidences:
        return (
            f"{display_name}未检索到相似度不低于 {SIMILARITY_THRESHOLD:.1f} "
            "的有效证据，当前问题不能据此给出确定答案。",
            [],
        )

    lines = [f"根据{display_name}检索到的高相关证据："]
    for evidence in evidences:
        label = evidence.get("standard_id") or f"S{evidence['rank']}"
        lines.append(
            f"{evidence['rank']}. [{label} | score={evidence['score']:.4f}] "
            f"{answer_excerpt(selected[evidence['rank'] - 1])}"
        )
    return "\n".join(lines), evidences


class RagflowSearchClient:
    def __init__(self, base_url: str, api_key: str, timeout: float) -> None:
        self.base_url = normalize_base_url(base_url)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

    def search(
        self,
        *,
        dataset_id: str,
        question: str,
        size: int,
        top_k: int,
        vector_weight: float,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": question,
            "doc_ids": [],
            "page": 1,
            "size": size,
            "top_k": top_k,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "vector_similarity_weight": vector_weight,
            "use_kg": False,
            "keyword": False,
        }
        return self._post_search(dataset_id, payload)

    def _post_search(
        self, dataset_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        url = f"{self.base_url}/datasets/{dataset_id}/search"
        try:
            response = self.session.post(url, json=dict(payload), timeout=self.timeout)
            response.raise_for_status()
        except requests.Timeout as exc:
            raise RuntimeError(f"RAGFlow 请求超时：{url}") from exc
        except requests.HTTPError as exc:
            detail = response.text[:500]
            if response.status_code in (401, 403):
                raise RuntimeError(
                    "RAGFlow 鉴权失败，请检查 RAGFLOW_PUBLIC_API_KEY。"
                ) from exc
            raise RuntimeError(
                f"RAGFlow HTTP {response.status_code}: {detail}"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"无法连接 RAGFlow：{exc}") from exc

        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError(f"RAGFlow 返回的不是 JSON：{response.text[:500]}") from exc
        if not isinstance(body, dict):
            raise RuntimeError(f"RAGFlow 响应格式异常：{body!r}")
        if body.get("code") not in (None, 0, "0", "ok", "OK"):
            raise RuntimeError(str(body.get("message") or body))
        data = body.get("data", body)
        if not isinstance(data, dict):
            raise RuntimeError(f"RAGFlow data 字段格式异常：{data!r}")
        return data


def build_parser(profile: DatasetProfile, base_url: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"独立检索 {profile.display_name}，固定相似度阈值 "
            f"{SIMILARITY_THRESHOLD:.1f}，按相关性降序返回答案和证据。"
        )
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--question", help="只运行一个自定义问题")
    selection.add_argument(
        "--list", action="store_true", help="只显示数据集配置和内置问题，不连接 RAGFlow"
    )
    parser.add_argument("--base-url", default=os.environ.get("RAGFLOW_BASE_URL", base_url))
    parser.add_argument("--dataset-id", help="临时覆盖 manifest.json 中的数据集 ID")
    parser.add_argument("--document-id", help="临时覆盖 manifest.json 中的文档 ID")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--answer-limit", type=int, default=DEFAULT_ANSWER_LIMIT)
    parser.add_argument("--vector-weight", type=float, default=DEFAULT_VECTOR_WEIGHT)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.size < 1 or args.top_k < 1 or args.answer_limit < 1:
        raise ValueError("--size、--top-k 和 --answer-limit 必须大于等于 1")
    if not 0 <= args.vector_weight <= 1:
        raise ValueError("--vector-weight 必须在 0 到 1 之间")
    if args.timeout <= 0:
        raise ValueError("--timeout 必须大于 0")


def run_dataset_cli(profile: DatasetProfile) -> int:
    configure_console()
    try:
        manifest = load_manifest()
        entry = find_manifest_dataset(manifest, profile.dataset_name)
        default_base_url = str(
            manifest.get("ragflow_base_url") or "http://localhost:9380/api/v1"
        )
        args = build_parser(profile, default_base_url).parse_args()
        validate_args(args)
    except (RuntimeError, ValueError) as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    dataset_id = str(args.dataset_id or entry.get("dataset_id") or "").strip()
    document_id = str(args.document_id or entry.get("document_id") or "").strip()
    if not dataset_id:
        print(f"配置错误：{profile.dataset_name} 缺少 dataset_id", file=sys.stderr)
        return 2

    if args.list:
        print(f"公共库：{profile.display_name}")
        print(f"dataset_name：{profile.dataset_name}")
        print(f"dataset_id：{dataset_id}")
        print(f"document_id：{document_id or '未限定'}")
        print(f"索引文档：{profile.indexed_document_name}")
        print(f"相似度阈值：{SIMILARITY_THRESHOLD}")
        for index, question in enumerate(profile.questions, 1):
            print(f"Q{index}: {question}")
        return 0

    api_key = normalize_api_key(
        os.environ.get("RAGFLOW_PUBLIC_API_KEY") or os.environ.get("RAGFLOW_API_KEY")
    )
    if not api_key:
        print(
            "缺少 RAGFLOW_PUBLIC_API_KEY（或 RAGFLOW_API_KEY）环境变量。",
            file=sys.stderr,
        )
        return 2

    questions = (args.question,) if args.question else profile.questions
    client = RagflowSearchClient(args.base_url, api_key, args.timeout)
    results: list[dict[str, Any]] = []
    failures = 0
    for question in questions:
        try:
            data = client.search(
                dataset_id=dataset_id,
                question=question,
                size=args.size,
                top_k=args.top_k,
                vector_weight=args.vector_weight,
            )
            retrieved = extract_chunks(data)
            scoped = filter_document_chunks(
                retrieved,
                document_id or None,
                profile.indexed_document_name,
            )
            ranked = rank_chunks(scoped)
            answer, evidences = build_answer(
                profile.display_name, ranked, args.answer_limit
            )
            results.append(
                {
                    "dataset_key": profile.key,
                    "dataset_name": profile.dataset_name,
                    "dataset_id": dataset_id,
                    "document_id": document_id or None,
                    "indexed_document_name": profile.indexed_document_name,
                    "question": question,
                    "similarity_threshold": SIMILARITY_THRESHOLD,
                    "retrieved_count": len(retrieved),
                    "scoped_count": len(scoped),
                    "accepted_count": len(ranked),
                    "answer": answer,
                    "evidences": evidences,
                }
            )
            if not args.json:
                print(f"\n问题：{question}\n\n{answer}")
        except RuntimeError as exc:
            failures += 1
            results.append(
                {
                    "dataset_key": profile.key,
                    "dataset_name": profile.dataset_name,
                    "question": question,
                    "error": str(exc),
                }
            )
            if not args.json:
                print(f"\n问题：{question}\n错误：{exc}", file=sys.stderr)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    return 1 if failures else 0
