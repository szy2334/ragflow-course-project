"""RAGFlow 五个论文知识库的逐库检索测试工具。

这个脚本直接调用 ``POST /api/v1/datasets/{dataset_id}/search``，每个知识库
使用自己的相似度阈值。它不会把 API Key 写入文件，也不会修改 RAGFlow Chat。

推荐用法（PowerShell）：

    $env:RAGFLOW_API_KEY = "你的 API Key"
    python public_kb/ragflow_retrieval_test.py --list
    python public_kb/ragflow_retrieval_test.py --kb icml_content
    python public_kb/ragflow_retrieval_test.py --kb user_paper --question "去掉L_gen后结果下降多少？"
    python public_kb/ragflow_retrieval_test.py --all

如果公共知识库和用户论文库属于不同 RAGFlow 租户，可分别设置：

    RAGFLOW_PUBLIC_API_KEY
    RAGFLOW_USER_API_KEY

二者未设置时都会回退到 RAGFLOW_API_KEY。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import requests


DEFAULT_BASE_URL = "http://localhost/api/v1"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    threshold: float
    key_scope: str
    expected_documents: tuple[str, ...]
    questions: tuple[str, ...]


# 顺序与约定一致：学位内容、ICML 内容、NeurIPS 格式、用户论文、学位格式。
DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="degree_content",
        name="public_degree_content_2026",
        threshold=0.2,
        key_scope="public",
        expected_documents=("degree_content_source_2026.pdf",),
        questions=(
            "学位论文摘要需要包含哪些内容？",
            "如何评价学位论文的论点是否明确、论证是否充分？",
            "学位论文综合评价分为哪些等级？",
        ),
    ),
    DatasetSpec(
        key="icml_content",
        name="public_research_content_icml_2026",
        threshold=0.2,
        key_scope="public",
        expected_documents=("icml_2026_reviewer_instructions.html",),
        questions=(
            "ICML如何评价论文的技术可靠性？",
            "如何判断实验是否充分支持论文的核心贡献？",
            "Soundness、Presentation、Significance和Originality如何评分？",
        ),
    ),
    DatasetSpec(
        key="neurips_format",
        name="public_research_format_neurips_2026",
        threshold=0.2,
        key_scope="public",
        expected_documents=("neurips_2026_format_source.txt",),
        questions=(
            "NeurIPS论文主PDF中的内容顺序是什么？",
            "NeurIPS论文是否需要包含论文检查清单？",
            "请只检查NeurIPS格式，不评价论文内容。",
        ),
    ),
    DatasetSpec(
        key="user_paper",
        name="user_papers_private_v2_strict_ocr",
        threshold=0.2,
        key_scope="user",
        expected_documents=("UMAP.pdf",),
        questions=(
            "这篇论文使用了哪些数据集和数据模态？",
            "论文是否进行了完整模态、仅EEG和仅眼动实验？",
            "去掉L_gen后，SEED-V上的实验结果下降了多少？",
        ),
    ),
    DatasetSpec(
        key="degree_format",
        name="public_degree_format_2026",
        threshold=0.2,
        key_scope="public",
        expected_documents=("degree_format_document.txt",),
        questions=(
            "学位论文中的英文缩写第一次出现时应该怎么写？",
            "学位论文的标点符号和专业术语有哪些要求？",
            "请只检查学位论文格式，不评价论文研究内容。",
        ),
    ),
)


def configure_console() -> None:
    """尽量保持中文输入输出，不强制修改真实终端的代码页。"""

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
        return DEFAULT_BASE_URL
    if not base_url.endswith("/api/v1"):
        base_url += "/api/v1"
    return base_url


class RagflowClient:
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

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                timeout=self.timeout,
                **kwargs,
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise RuntimeError(f"RAGFlow 请求超时：{self.base_url}{path}") from exc
        except requests.HTTPError as exc:
            detail = response.text[:500]
            if response.status_code in (401, 403):
                raise RuntimeError(
                    f"RAGFlow 鉴权失败（HTTP {response.status_code}），请检查对应租户的 API Key。"
                ) from exc
            raise RuntimeError(
                f"RAGFlow HTTP {response.status_code}: {detail}"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"无法连接 RAGFlow：{exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"RAGFlow 返回的不是 JSON：{response.text[:500]}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"RAGFlow 响应格式异常：{payload!r}")
        if payload.get("code") not in (None, 0, "0", "ok", "OK"):
            raise RuntimeError(str(payload.get("message") or payload))
        return payload.get("data", payload)

    def list_datasets(self) -> list[dict[str, Any]]:
        data = self.request("GET", "/datasets", params={"page": 1, "page_size": 100})
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            for key in ("datasets", "items", "records", "data"):
                items = data.get(key)
                if isinstance(items, list):
                    return [item for item in items if isinstance(item, dict)]
        raise RuntimeError(f"无法识别知识库列表响应：{data!r}")

    def search(
        self,
        dataset_id: str,
        question: str,
        threshold: float,
        size: int,
        vector_similarity_weight: float,
    ) -> dict[str, Any]:
        data = self.request(
            "POST",
            f"/datasets/{dataset_id}/search",
            json={
                "question": question,
                "doc_ids": [],
                "page": 1,
                "size": size,
                "top_k": 1024,
                "similarity_threshold": threshold,
                "vector_similarity_weight": vector_similarity_weight,
                "use_kg": False,
                "keyword": False,
            },
        )
        if not isinstance(data, dict):
            raise RuntimeError(f"检索响应格式异常：{data!r}")
        return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按知识库独立阈值测试 RAGFlow 检索结果"
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("RAGFLOW_BASE_URL", DEFAULT_BASE_URL),
        help="RAGFlow 地址，默认 http://localhost/api/v1",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--kb",
        help="知识库短名或完整名称，例如 icml_content 或 public_research_content_icml_2026",
    )
    selection.add_argument("--all", action="store_true", help="运行全部内置问题")
    selection.add_argument("--list", action="store_true", help="仅列出配置和内置问题")
    parser.add_argument("--question", help="自定义中文问题；必须同时提供 --kb")
    parser.add_argument("--threshold", type=float, help="临时覆盖所选知识库阈值")
    parser.add_argument("--size", type=int, default=5, help="每个问题最多显示的 Chunk 数")
    parser.add_argument(
        "--vector-weight",
        type=float,
        default=0.3,
        help="向量相似度权重，默认 0.3",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--dataset-id",
        action="append",
        default=[],
        metavar="KEY=ID",
        help="知识库 ID 覆盖，可重复提供，例如 --dataset-id user_paper=xxxx",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出，便于自动化处理")
    return parser.parse_args()


def parse_dataset_overrides(values: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, dataset_id = value.partition("=")
        if not separator or not key.strip() or not dataset_id.strip():
            raise ValueError(f"无效 --dataset-id：{value!r}，正确格式为 KEY=ID")
        result[key.strip()] = dataset_id.strip()
    return result


def find_spec(value: str) -> DatasetSpec:
    normalized = value.strip().lower()
    for spec in DATASETS:
        if normalized in {spec.key.lower(), spec.name.lower()}:
            return spec
    choices = ", ".join(spec.key for spec in DATASETS)
    raise ValueError(f"未知知识库 {value!r}；可选值：{choices}")


def print_catalog() -> None:
    print("五个知识库的检索测试配置：")
    for index, spec in enumerate(DATASETS, start=1):
        print(f"\n{index}. {spec.key} / {spec.name}")
        print(f"   相似度阈值：{spec.threshold}")
        print(f"   预期文档：{', '.join(spec.expected_documents)}")
        for question_index, question in enumerate(spec.questions, start=1):
            print(f"   Q{question_index}: {question}")


def make_clients(args: argparse.Namespace) -> dict[str, RagflowClient]:
    common_key = normalize_api_key(os.environ.get("RAGFLOW_API_KEY"))
    public_key = normalize_api_key(os.environ.get("RAGFLOW_PUBLIC_API_KEY")) or common_key
    user_key = normalize_api_key(os.environ.get("RAGFLOW_USER_API_KEY")) or common_key
    missing: list[str] = []
    if not public_key:
        missing.append("RAGFLOW_PUBLIC_API_KEY（或 RAGFLOW_API_KEY）")
    if not user_key:
        missing.append("RAGFLOW_USER_API_KEY（或 RAGFLOW_API_KEY）")
    if missing:
        raise RuntimeError("缺少环境变量：" + "、".join(missing))
    return {
        "public": RagflowClient(args.base_url, public_key, args.timeout),
        "user": RagflowClient(args.base_url, user_key, args.timeout),
    }


def resolve_dataset_ids(
    specs: Iterable[DatasetSpec],
    clients: Mapping[str, RagflowClient],
    overrides: Mapping[str, str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    cache: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        override = overrides.get(spec.key) or overrides.get(spec.name)
        if override:
            result[spec.key] = override
            continue
        if spec.key_scope not in cache:
            cache[spec.key_scope] = clients[spec.key_scope].list_datasets()
        matches = [
            item
            for item in cache[spec.key_scope]
            if str(item.get("name") or item.get("dataset_name") or "") == spec.name
        ]
        if not matches:
            raise RuntimeError(
                f"当前 {spec.key_scope} API Key 看不到知识库 {spec.name}。"
                f"请检查租户/API Key，或使用 --dataset-id {spec.key}=实际ID。"
            )
        dataset_id = matches[0].get("id") or matches[0].get("dataset_id")
        if not dataset_id:
            raise RuntimeError(f"知识库 {spec.name} 没有返回 ID：{matches[0]!r}")
        result[spec.key] = str(dataset_id)
    return result


def extract_chunks(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    chunks = data.get("chunks") or data.get("chunk") or []
    if not isinstance(chunks, list):
        return []
    return [item for item in chunks if isinstance(item, dict)]


def chunk_summary(chunk: Mapping[str, Any]) -> dict[str, Any]:
    content = str(
        chunk.get("content_with_weight")
        or chunk.get("content")
        or chunk.get("text")
        or chunk.get("content_ltks")
        or ""
    ).strip()
    return {
        "document": chunk.get("document_name")
        or chunk.get("doc_name")
        or chunk.get("docnm_kwd")
        or chunk.get("document_id")
        or chunk.get("doc_id")
        or "未知文档",
        "similarity": chunk.get("similarity", chunk.get("score")),
        "content": content,
    }


def run_tests(args: argparse.Namespace) -> int:
    if args.list or (not args.kb and not args.all):
        print_catalog()
        if not args.kb and not args.all:
            print("\n使用 --kb 运行单库测试，或使用 --all 运行全部测试。")
        return 0
    if args.question and not args.kb:
        raise ValueError("--question 必须与 --kb 一起使用")
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        raise ValueError("--threshold 必须在 0 到 1 之间")
    if not 0 <= args.vector_weight <= 1:
        raise ValueError("--vector-weight 必须在 0 到 1 之间")
    if args.size < 1:
        raise ValueError("--size 必须大于等于 1")

    specs = DATASETS if args.all else (find_spec(args.kb),)
    clients = make_clients(args)
    overrides = parse_dataset_overrides(args.dataset_id)
    dataset_ids = resolve_dataset_ids(specs, clients, overrides)
    output: list[dict[str, Any]] = []
    failures = 0

    for spec in specs:
        threshold = args.threshold if args.threshold is not None else spec.threshold
        questions = (args.question,) if args.question else spec.questions
        if not args.json:
            print(f"\n=== {spec.name} | threshold={threshold} ===")
        for question in questions:
            try:
                data = clients[spec.key_scope].search(
                    dataset_ids[spec.key],
                    question,
                    threshold,
                    args.size,
                    args.vector_weight,
                )
                chunks = [chunk_summary(item) for item in extract_chunks(data)]
                record = {
                    "key": spec.key,
                    "dataset_name": spec.name,
                    "dataset_id": dataset_ids[spec.key],
                    "threshold": threshold,
                    "question": question,
                    "hit_count": len(chunks),
                    "chunks": chunks,
                }
                output.append(record)
                if not args.json:
                    print(f"\n问题：{question}")
                    if not chunks:
                        print("结果：无命中")
                    for index, chunk in enumerate(chunks, start=1):
                        content = str(chunk["content"]).replace("\n", " ")
                        if len(content) > 260:
                            content = content[:260] + "..."
                        print(
                            f"[R{index}] {chunk['document']}  similarity={chunk['similarity']}"
                        )
                        print(f"     {content}")
            except RuntimeError as exc:
                failures += 1
                output.append(
                    {
                        "key": spec.key,
                        "dataset_name": spec.name,
                        "dataset_id": dataset_ids.get(spec.key),
                        "threshold": threshold,
                        "question": question,
                        "error": str(exc),
                    }
                )
                if not args.json:
                    print(f"\n问题：{question}\n错误：{exc}", file=sys.stderr)

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    return 1 if failures else 0


def main() -> int:
    configure_console()
    args = parse_args()
    try:
        return run_tests(args)
    except (RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
