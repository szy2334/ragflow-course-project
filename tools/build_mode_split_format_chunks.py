"""Build mode-scoped format-rule JSONL drafts without changing source files."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path


NAMESPACE = uuid.UUID("924ec1d6-00d5-4da1-aa24-bb6dc1397a7a")

ICML_INITIAL = {
    "d5289680-0d1b-5a36-a698-c214814b3974",
    "ee882f2b-da3e-532c-acaf-c07109b8e033",
    "cacf61bf-35a6-5dd0-86a9-d095fd59e93e",
}
ICML_CAMERA = {
    "7b2f2f76-656b-5656-8ead-c9ddfc0d2f61",
    "25f9cb76-f69a-5f0d-adbb-59fb57fc731a",
    "2bdfaf57-95d1-5fb9-8604-1b3e67222992",
}
ICML_MIXED = {
    "2dff05d5-5ed5-5507-85a2-7ca62f1ca962",
    "351d3f1c-8942-54f9-9ef0-1b660d70b16a",
    "0609c1ca-6d48-58ab-8c30-060990471870",
    "5ac4d355-32f8-53ea-b61b-9127cf9374df",
}

NEURIPS_INITIAL = {
    "2ff08d54-0cae-504d-ab06-3ee8eb51fed6",
    "cdb3abf6-fa66-5cfc-b272-9524f44096ae",
}
NEURIPS_CAMERA = {"8d106e53-e532-5c20-93c5-3c014f4fb153"}
NEURIPS_MIXED = {
    "5a670881-8c27-5bf8-bdce-d7a2df992a01",
    "c5632e21-d781-50f4-bd70-3573b9176037",
    "974d1491-ed8a-5ec0-893d-4fc4c25af118",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def rule_id(venue_id: str, name: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{venue_id}:{name}"))


def category(section: str) -> str:
    value = section.lower()
    if "author" in value or "anonymous" in value:
        return "author"
    if "abstract" in value:
        return "abstract"
    if "figure" in value:
        return "figure"
    if "table" in value:
        return "table"
    if "reference" in value or "citation" in value:
        return "reference"
    if "appendix" in value:
        return "appendix"
    if "heading" in value or "section" in value:
        return "heading"
    if "dimension" in value or "margin" in value or "page" in value:
        return "page"
    return "body"


def normalized_copy(
    source: dict,
    *,
    venue_id: str,
    format_version: str,
    target_document: str,
    submission_mode: str,
) -> dict:
    metadata = dict(source["metadata"])
    metadata.update(
        {
            "venue_id": venue_id,
            "format_version": format_version,
            "target_document": target_document,
            "submission_mode": submission_mode,
            "canonical_rule_id": rule_id(
                venue_id, f"{submission_mode}:{source['document_id']}"
            ),
            "rule_category": category(str(metadata.get("section", ""))),
            "source_document_id": source["document_id"],
            "normalization": "mode_split_v1",
        }
    )
    content = "\n".join(
        [
            f"规则文档：{target_document}",
            f"投稿模式：{submission_mode}",
            source["content"],
        ]
    )
    return {
        "document_id": rule_id(venue_id, f"{submission_mode}:{source['document_id']}"),
        "content": content,
        "metadata": metadata,
    }


def derived_rule(
    source: dict,
    *,
    venue_id: str,
    format_version: str,
    target_document: str,
    submission_mode: str,
    name: str,
    rule_category: str,
    text: str,
) -> dict:
    source_meta = source["metadata"]
    content = "\n".join(
        [
            f"格式参考：{source_meta['venue']}",
            f"年份：{source_meta['venue_year']}",
            f"投稿模式：{submission_mode}",
            f"章节：{source_meta['section']}",
            f"来源页码：{source_meta['page_start']}-{source_meta['page_end']}",
            "清洗后的规则原文：",
            text,
        ]
    )
    identifier = rule_id(venue_id, name)
    return {
        "document_id": identifier,
        "content": content,
        "metadata": {
            **source_meta,
            "venue_id": venue_id,
            "format_version": format_version,
            "target_document": target_document,
            "submission_mode": submission_mode,
            "canonical_rule_id": identifier,
            "rule_category": rule_category,
            "source_document_id": source["document_id"],
            "normalization": "mode_split_v1",
            "derived_from_mixed_source": True,
        },
    }


def by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["document_id"]: row for row in rows}


def split_icml(rows: list[dict]) -> list[dict]:
    sources = by_id(rows)
    output: list[dict] = []
    for row in rows:
        source_id = row["document_id"]
        if source_id in ICML_MIXED:
            continue
        if source_id in ICML_INITIAL:
            mode, document = "initial_submission", "icml_2026_initial_submission_rules"
        elif source_id in ICML_CAMERA:
            mode, document = "camera_ready", "icml_2026_camera_ready_rules"
        else:
            mode, document = "shared", "icml_2026_shared_rules"
        output.append(
            normalized_copy(
                row,
                venue_id="icml",
                format_version="2026",
                target_document=document,
                submission_mode=mode,
            )
        )

    output.extend(
        [
            derived_rule(sources["2dff05d5-5ed5-5507-85a2-7ca62f1ca962"], venue_id="icml", format_version="2026", target_document="icml_2026_shared_rules", submission_mode="shared", name="submission-basics", rule_category="page", text="Submissions must be in PDF. Appendices, the main body, and references must be submitted together as a single file. Use 10 point Times font and embedded Type-1 fonts. Place figure captions under figures and table captions over tables. Do not alter the style template or compress vertical spacing. Keep the abstract to one self-contained paragraph of roughly 4-6 sentences, and capitalize content words in the title."),
            derived_rule(sources["2dff05d5-5ed5-5507-85a2-7ca62f1ca962"], venue_id="icml", format_version="2026", target_document="icml_2026_initial_submission_rules", submission_mode="initial_submission", name="initial-page-limit", rule_category="page", text="For an initial submission, the main body must fit within 8 pages, excluding references and appendices; the total PDF file size must not exceed 10MB."),
            derived_rule(sources["2dff05d5-5ed5-5507-85a2-7ca62f1ca962"], venue_id="icml", format_version="2026", target_document="icml_2026_initial_submission_rules", submission_mode="initial_submission", name="initial-anonymity-summary", rule_category="author", text="Do not include author information or acknowledgements in an initial submission."),
            derived_rule(sources["2dff05d5-5ed5-5507-85a2-7ca62f1ca962"], venue_id="icml", format_version="2026", target_document="icml_2026_camera_ready_rules", submission_mode="camera_ready", name="camera-ready-page-limit", rule_category="page", text="For the final camera-ready version, authors may add one extra page to the main body beyond the initial 8-page limit."),
            derived_rule(sources["351d3f1c-8942-54f9-9ef0-1b660d70b16a"], venue_id="icml", format_version="2026", target_document="icml_2026_shared_rules", submission_mode="shared", name="dimensions-shared", rule_category="page", text="Format the text in two columns with overall width 6.75 inches, height 9.0 inches, and 0.25 inches between columns. Use a 0.75-inch left margin, a 1.0-inch top margin, 10-point type with 11-point vertical spacing, and Times typeface. Do not write in the margins."),
            derived_rule(sources["351d3f1c-8942-54f9-9ef0-1b660d70b16a"], venue_id="icml", format_version="2026", target_document="icml_2026_camera_ready_rules", submission_mode="camera_ready", name="camera-ready-paper-size", rule_category="page", text="All final camera-ready versions must be produced for US letter size."),
            derived_rule(sources["0609c1ca-6d48-58ab-8c30-060990471870"], venue_id="icml", format_version="2026", target_document="icml_2026_initial_submission_rules", submission_mode="initial_submission", name="initial-acknowledgements", rule_category="author", text="Do not include acknowledgements in the initial version submitted for blind review."),
            derived_rule(sources["0609c1ca-6d48-58ab-8c30-060990471870"], venue_id="icml", format_version="2026", target_document="icml_2026_camera_ready_rules", submission_mode="camera_ready", name="camera-ready-acknowledgements", rule_category="body", text="A final camera-ready version may include acknowledgements at the end of the paper in an unnumbered section that does not count toward the page limit."),
            derived_rule(sources["5ac4d355-32f8-53ea-b61b-9127cf9374df"], venue_id="icml", format_version="2026", target_document="icml_2026_shared_rules", submission_mode="shared", name="appendix-format", rule_category="appendix", text="An appendix may use one or two columns, but its font size, spacing, margins, and page numbering should otherwise remain the same as the main body."),
            derived_rule(sources["5ac4d355-32f8-53ea-b61b-9127cf9374df"], venue_id="icml", format_version="2026", target_document="icml_2026_initial_submission_rules", submission_mode="initial_submission", name="initial-appendix-page-limit", rule_category="page", text="For an initial submission, the main body must be at most 8 pages; appendix text is not limited."),
            derived_rule(sources["5ac4d355-32f8-53ea-b61b-9127cf9374df"], venue_id="icml", format_version="2026", target_document="icml_2026_camera_ready_rules", submission_mode="camera_ready", name="camera-ready-appendix-page-limit", rule_category="page", text="For a camera-ready version, the main body may include one more page than the initial 8-page limit; appendix text is not limited."),
        ]
    )
    return output


def split_neurips(rows: list[dict]) -> list[dict]:
    sources = by_id(rows)
    output: list[dict] = []
    for row in rows:
        source_id = row["document_id"]
        if source_id in NEURIPS_MIXED:
            continue
        if source_id in NEURIPS_INITIAL:
            mode, document = "initial_submission", "neurips_2020_initial_submission_rules"
        elif source_id in NEURIPS_CAMERA:
            mode, document = "camera_ready", "neurips_2020_camera_ready_rules"
        else:
            mode, document = "shared", "neurips_2020_shared_rules"
        output.append(
            normalized_copy(
                row,
                venue_id="neurips",
                format_version="2020",
                target_document=document,
                submission_mode=mode,
            )
        )

    output.extend(
        [
            derived_rule(sources["5a670881-8c27-5bf8-bdce-d7a2df992a01"], venue_id="neurips", format_version="2020", target_document="neurips_2020_shared_rules", submission_mode="shared", name="style-file", rule_category="body", text="Use the supported neurips_2020.sty LaTeX style file. Previous LaTeX 2.09, Microsoft Word, and RTF style files are not supported."),
            derived_rule(sources["5a670881-8c27-5bf8-bdce-d7a2df992a01"], venue_id="neurips", format_version="2020", target_document="neurips_2020_initial_submission_rules", submission_mode="initial_submission", name="initial-submission-mode", rule_category="author", text="At submission time, omit the final and preprint options. This anonymizes the submission and adds line numbers for review."),
            derived_rule(sources["5a670881-8c27-5bf8-bdce-d7a2df992a01"], venue_id="neurips", format_version="2020", target_document="neurips_2020_camera_ready_rules", submission_mode="camera_ready", name="camera-ready-mode", rule_category="author", text="Use the final option only for papers accepted to NeurIPS; it creates the camera-ready copy."),
            derived_rule(sources["5a670881-8c27-5bf8-bdce-d7a2df992a01"], venue_id="neurips", format_version="2020", target_document="neurips_2020_preprint_rules", submission_mode="preprint", name="preprint-mode", rule_category="author", text="Use the preprint option for a nonanonymized preprint. It adds the footer 'Preprint. Work in progress.' and may be distributed."),
            derived_rule(sources["c5632e21-d781-50f4-bd70-3573b9176037"], venue_id="neurips", format_version="2020", target_document="neurips_2020_shared_rules", submission_mode="shared", name="general-layout", rule_category="page", text="Confine text within a 5.5-inch-wide and 9-inch-long rectangle with a 1.5-inch left margin. Use 10-point type with 11-point leading, Times New Roman, paragraphs separated by 5.5 points without indentation, and a 17-point bold centered title between horizontal rules. All pages start 1 inch from the top."),
            derived_rule(sources["c5632e21-d781-50f4-bd70-3573b9176037"], venue_id="neurips", format_version="2020", target_document="neurips_2020_camera_ready_rules", submission_mode="camera_ready", name="camera-ready-author-layout", rule_category="author", text="For the final version, authors' names are bold and centered above corresponding addresses. List the lead author first and arrange co-authors according to the stated address rules."),
            derived_rule(sources["974d1491-ed8a-5ec0-893d-4fc4c25af118"], venue_id="neurips", format_version="2020", target_document="neurips_2020_initial_submission_rules", submission_mode="initial_submission", name="initial-acknowledgements", rule_category="author", text="Do not include acknowledgments, funding disclosure, or competing-interest disclosure in the anonymized submission."),
            derived_rule(sources["974d1491-ed8a-5ec0-893d-4fc4c25af118"], venue_id="neurips", format_version="2020", target_document="neurips_2020_camera_ready_rules", submission_mode="camera_ready", name="camera-ready-acknowledgements", rule_category="body", text="In the final paper, use an unnumbered first-level acknowledgments section before references and include required funding and competing-interest disclosures."),
        ]
    )
    return output


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def summarize(rows: list[dict]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        key = str(row["metadata"]["target_document"])
        result[key] = result.get(key, 0) + 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--icml-source", type=Path, required=True)
    parser.add_argument("--neurips-source", type=Path, required=True)
    parser.add_argument("--icml-output", type=Path, required=True)
    parser.add_argument("--neurips-output", type=Path, required=True)
    args = parser.parse_args()

    icml = split_icml(read_jsonl(args.icml_source))
    neurips = split_neurips(read_jsonl(args.neurips_source))
    write_jsonl(args.icml_output, icml)
    write_jsonl(args.neurips_output, neurips)
    print(json.dumps({"icml": summarize(icml), "neurips": summarize(neurips)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
