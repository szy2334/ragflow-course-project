"""Evaluate a fixed UMAP paper QA golden baseline.

This utility separates retrieval correctness from answer correctness. It consumes
RAGFlow-style QA result JSON files and the source-to-RAGFlow chunk mapping, so it
can be used as a regression gate without exposing any runtime credentials. The
``answerable`` field is descriptive metadata only in this related-question
baseline and does not affect scoring.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pipeline_common import config_bool, config_value, load_user_paper_config


load_user_paper_config()


REFUSAL_MARKERS = (
    "the paper does not provide this information",
    "the answer you are looking for is not found in the dataset",
    "\u8bba\u6587\u672a\u63d0\u4f9b",
    "\u8bba\u6587\u4e2d\u672a\u627e\u5230\u8db3\u591f\u4fe1\u606f",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize(value: str | None) -> str:
    value = value or ""
    return re.sub(r"\s+", "", value).lower()


def contains_any(answer: str, alternatives: list[str]) -> bool:
    answer_norm = normalize(answer)
    return any(normalize(item) in answer_norm for item in alternatives)


def is_refusal(answer: str) -> bool:
    answer_norm = normalize(answer)
    return any(normalize(marker) in answer_norm for marker in REFUSAL_MARKERS)


def source_to_ragflow(mapping_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in iter_jsonl(mapping_path):
        source_id = str(row.get("source_chunk_id") or "")
        ragflow_id = str(row.get("ragflow_chunk_id") or "")
        if source_id and ragflow_id:
            mapping[source_id] = ragflow_id
    return mapping


def find_result(results: list[dict[str, Any]], case: dict[str, Any]) -> dict[str, Any] | None:
    case_id = str(case.get("id") or "")
    if case_id:
        for result in results:
            if str(result.get("id") or "") == case_id:
                return result
    for result in results:
        if normalize(result.get("question")) == normalize(case.get("question_zh")):
            return result
    return None


def evaluate_case(
    case: dict[str, Any],
    result: dict[str, Any] | None,
    mapping: dict[str, str],
    *,
    retrieval_only: bool = False,
) -> dict[str, Any]:
    expected_sources = list(case.get("expected_source_chunk_ids") or [])
    expected_ragflow_ids = {mapping[source] for source in expected_sources if source in mapping}
    report: dict[str, Any] = {
        "id": case["id"],
        "kind": case.get("result_kind"),
        "question_zh": case["question_zh"],
        "answerable": bool(case["answerable"]),
        "expected_source_chunk_ids": expected_sources,
        "expected_ragflow_chunk_ids": sorted(expected_ragflow_ids),
    }
    if result is None:
        report.update({
            "status": "NOT_RUN",
            "retrieval_pass": False,
            "answer_pass": False,
            "citation_pass": False,
            "failure_reasons": ["No matching test result was found."],
        })
        return report

    answer = str(result.get("answer") or "")
    references = result.get("references") or []
    actual_ragflow_ids = {str(item.get("id")) for item in references if item.get("id")}
    citation_pass = True if not expected_ragflow_ids else bool(actual_ragflow_ids & expected_ragflow_ids)
    # A retrieval service returns nearest neighbours. Refusal is owned by the
    # answer agent, so only configured evidence expectations are scored here.
    retrieval_pass = citation_pass
    failure_reasons: list[str] = []

    if retrieval_only:
        if not citation_pass:
            failure_reasons.append("No retrieved citation matches a golden evidence chunk.")
        status = "PASS" if retrieval_pass and citation_pass else "FAIL"
        report.update({
            "status": status,
            "answer": None,
            "reference_count": len(references),
            "actual_ragflow_chunk_ids": sorted(actual_ragflow_ids),
            "retrieval_pass": retrieval_pass,
            "citation_pass": citation_pass,
            "answer_pass": None,
            "failure_reasons": failure_reasons,
        })
        return report

    groups = case.get("expected_fact_groups") or []
    numbers = case.get("expected_numbers") or []
    forbidden_substrings = list(case.get("forbidden_answer_substrings") or [])
    has_answer_assertions = bool(groups or numbers or forbidden_substrings)
    if has_answer_assertions:
        missing_groups = [group for group in groups if not contains_any(answer, list(group))]
        missing_numbers = [value for value in numbers if normalize(value) not in normalize(answer)]
        answer_pass = bool(answer.strip()) and not missing_groups and not missing_numbers
        if missing_groups:
            failure_reasons.append(f"Missing expected fact groups: {missing_groups}")
        if missing_numbers:
            failure_reasons.append(f"Missing or altered exact numeric values: {missing_numbers}")
        matched_forbidden = [item for item in forbidden_substrings if normalize(item) in normalize(answer)]
        if matched_forbidden:
            answer_pass = False
            failure_reasons.append(f"The answer contains forbidden or contradictory output: {matched_forbidden}")
    else:
        answer_pass = True

    if not citation_pass:
        failure_reasons.append("No retrieved citation matches a golden evidence chunk.")

    status = "PASS" if answer_pass and citation_pass and retrieval_pass else "FAIL"
    report.update({
        "status": status,
        "answer": answer,
        "reference_count": len(references),
        "actual_ragflow_chunk_ids": sorted(actual_ragflow_ids),
        "retrieval_pass": retrieval_pass,
        "citation_pass": citation_pass,
        "answer_pass": answer_pass,
        "failure_reasons": failure_reasons,
    })
    return report


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# UMAP \u9ec4\u91d1\u95ee\u7b54\u57fa\u7ebf\u62a5\u544a",
        "",
        f"- \u8bba\u6587\uff1a`{report['paper']['document_name']}`",
        f"- \u6587\u4ef6\u54c8\u5e0c\uff1a`{report['paper']['file_sha256']}`",
        f"- \u6d4b\u8bd5\u7ed3\u679c\uff1a`{report['results_file']}`",
        "",
        "## \u6c47\u603b",
        "",
        f"- \u7528\u4f8b\u603b\u6570\uff1a{summary['total']}",
        f"- \u5df2\u6267\u884c\uff1a{summary['executed']}",
        f"- \u901a\u8fc7\uff1a{summary['passed']}",
        f"- \u5931\u8d25\uff1a{summary['failed']}",
        f"- \u672a\u6267\u884c\uff1a{summary['not_run']}",
        "",
        "| \u7528\u4f8b | \u72b6\u6001 | \u68c0\u7d22 | \u5f15\u7528 | \u56de\u7b54 | \u8bf4\u660e |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["cases"]:
        reasons = "\uff1b".join(item.get("failure_reasons") or []) or "\u7b26\u5408\u57fa\u7ebf"
        answer_pass = "N/A" if item["answer_pass"] is None else item["answer_pass"]
        lines.append(
            f"| {item['id']} | {item['status']} | {item['retrieval_pass']} | "
            f"{item['citation_pass']} | {answer_pass} | {reasons} |"
        )
    lines += [
        "",
        "## \u4f7f\u7528\u8bf4\u660e",
        "",
        "\u8be5\u57fa\u7ebf\u7528\u4e8e\u56de\u5f52\u6d4b\u8bd5\uff0c\u4e0d\u662f\u6a21\u578b\u8bad\u7ec3\u6570\u636e\u3002\u4fee\u6539\u6a21\u578b\u3001\u63d0\u793a\u8bcd\u3001\u68c0\u7d22\u7b56\u7565\u3001Chunk \u6216 RAGFlow \u914d\u7f6e\u540e\uff0c"
        "\u5747\u5e94\u91cd\u65b0\u6267\u884c\u672c\u8bc4\u4f30\uff0c\u5e76\u5c06\u7ed3\u679c\u4e0e\u672c\u62a5\u544a\u5bf9\u6bd4\u3002",
        "",
    ]
    return "\n".join(lines)


def configured_path(name: str) -> Path | None:
    value = config_value(name)
    return Path(value).expanduser() if value else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden",
        type=Path,
        default=configured_path("USER_PAPER_QA_GOLDEN"),
        help="Golden QA cases (default: USER_PAPER_QA_GOLDEN).",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=configured_path("USER_PAPER_QA_RESULTS"),
        help="RAGFlow-style QA results (default: USER_PAPER_QA_RESULTS).",
    )
    parser.add_argument(
        "--chunk-mapping",
        type=Path,
        default=configured_path("USER_PAPER_QA_CHUNK_MAPPING"),
        help="Source-to-RAGFlow mapping (default: USER_PAPER_QA_CHUNK_MAPPING).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=configured_path("USER_PAPER_QA_OUTPUT_JSON"),
        help="JSON report path (default: USER_PAPER_QA_OUTPUT_JSON).",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=configured_path("USER_PAPER_QA_OUTPUT_MD"),
        help="Markdown report path (default: USER_PAPER_QA_OUTPUT_MD).",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=config_bool("USER_PAPER_QA_STRICT", False),
        help="Return a non-zero status when any case fails or is not run.",
    )
    args = parser.parse_args()
    required = (
        ("--golden", "USER_PAPER_QA_GOLDEN", args.golden),
        ("--results", "USER_PAPER_QA_RESULTS", args.results),
        ("--chunk-mapping", "USER_PAPER_QA_CHUNK_MAPPING", args.chunk_mapping),
        ("--output-json", "USER_PAPER_QA_OUTPUT_JSON", args.output_json),
        ("--output-md", "USER_PAPER_QA_OUTPUT_MD", args.output_md),
    )
    missing = [
        f"{option} (set {setting} in .env)"
        for option, setting, value in required
        if value is None
    ]
    if missing:
        parser.error("Missing configuration: " + ", ".join(missing))
    return args


def main() -> int:
    args = parse_args()
    golden = read_json(args.golden)
    results_payload = read_json(args.results)
    mapping = source_to_ragflow(args.chunk_mapping)
    results = list(results_payload.get("results") or [])
    retrieval_only = results_payload.get("evaluation_mode") == "retrieval"
    cases = [
        evaluate_case(case, find_result(results, case), mapping, retrieval_only=retrieval_only)
        for case in golden["cases"]
    ]
    summary = {
        "total": len(cases),
        "executed": sum(case["status"] != "NOT_RUN" for case in cases),
        "passed": sum(case["status"] == "PASS" for case in cases),
        "failed": sum(case["status"] == "FAIL" for case in cases),
        "not_run": sum(case["status"] == "NOT_RUN" for case in cases),
    }
    report = {
        "schema_version": "user_paper_qa_baseline_report_v1",
        "golden_file": str(args.golden),
        "results_file": str(args.results),
        "evaluation_mode": "retrieval" if retrieval_only else "answer_and_retrieval",
        "paper": golden["paper"],
        "summary": summary,
        "cases": cases,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if args.strict and (summary["failed"] or summary["not_run"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
