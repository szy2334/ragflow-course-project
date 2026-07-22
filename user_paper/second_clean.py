"""Build source-of-truth and RAGFlow-ready chunks from MinerU and Baidu OCR."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from pipeline_common import (
    SCHEMA_VERSION,
    estimate_tokens,
    iter_jsonl,
    normalize_prose,
    normalize_text,
    quality_flags,
    read_json,
    sha256_text,
    stable_uuid,
    utc_now,
    write_json,
    write_jsonl,
)


CLEANING_VERSION = "paper_second_clean_v2"
_PREFERRED_TRUNCATION_BOUNDARIES = frozenset(
    "\n \t,;:.!?\u3002\uff0c\uff1b\uff1a\uff01\uff1f"
)


class HtmlTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int, int]]] = []
        self.current_row: list[tuple[str, int, int]] | None = None
        self.current_cell: list[str] | None = None
        self.rowspan = 1
        self.colspan = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.current_row = []
        elif tag in {"td", "th"} and self.current_row is not None:
            attributes = dict(attrs)
            self.current_cell = []
            self.rowspan = max(1, int(attributes.get("rowspan") or 1))
            self.colspan = max(1, int(attributes.get("colspan") or 1))
        elif tag == "br" and self.current_cell is not None:
            self.current_cell.append(" ")

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            self.current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.current_cell is not None:
            text = normalize_prose("".join(self.current_cell))
            assert self.current_row is not None
            self.current_row.append((text, self.rowspan, self.colspan))
            self.current_cell = None
        elif tag == "tr" and self.current_row is not None:
            self.rows.append(self.current_row)
            self.current_row = None


def html_table_to_matrix(table_html: str) -> list[list[str]]:
    if not table_html.strip():
        return []
    parser = HtmlTableParser()
    parser.feed(table_html)
    occupied: dict[tuple[int, int], str] = {}
    max_col = 0
    for row_index, cells in enumerate(parser.rows):
        column = 0
        for text, rowspan, colspan in cells:
            while (row_index, column) in occupied:
                column += 1
            for row_offset in range(rowspan):
                for col_offset in range(colspan):
                    occupied[(row_index + row_offset, column + col_offset)] = text
                    max_col = max(max_col, column + col_offset + 1)
            column += colspan
    max_row = max((row for row, _ in occupied), default=-1) + 1
    return [
        [occupied.get((row, column), "") for column in range(max_col)]
        for row in range(max_row)
    ]


def escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def matrix_to_markdown(matrix: list[list[str]]) -> str:
    if not matrix:
        return ""
    width = max(len(row) for row in matrix)
    rows = [row + [""] * (width - len(row)) for row in matrix]
    output = ["| " + " | ".join(escape_markdown_cell(v) for v in rows[0]) + " |"]
    output.append("| " + " | ".join("---" for _ in range(width)) + " |")
    for row in rows[1:]:
        output.append("| " + " | ".join(escape_markdown_cell(v) for v in row) + " |")
    return "\n".join(output)


def classify_section(section_path: list[str]) -> str:
    value = " ".join(section_path).lower()
    mappings = [
        ("abstract", ("abstract", "摘要")),
        ("introduction", ("introduction", "引言", "绪论")),
        ("related_work", ("related work", "background", "相关工作")),
        ("dataset", ("dataset", "data set", "数据集", "data and feature")),
        ("experiment", ("experiment", "implementation", "实验", "evaluation")),
        ("result", ("result", "comparison", "结果", "analysis")),
        ("ablation", ("ablation", "消融")),
        ("method", ("method", "approach", "model", "方法")),
        ("limitation", ("limitation", "future work", "局限", "不足")),
        ("conclusion", ("conclusion", "总结", "结论")),
        ("reference", ("reference", "bibliography", "参考文献")),
        ("appendix", ("appendix", "supplement", "附录")),
    ]
    for role, keywords in mappings:
        if any(keyword in value for keyword in keywords):
            return role
    return "body"


def section_label(path: list[str]) -> str:
    return " > ".join(part for part in path if part) or "Document"


def chunk_record(
    *,
    document: dict[str, Any],
    content_type: str,
    content_role: str,
    section_path: list[str],
    raw_content: str,
    embedding_text: str,
    page_start: int | None,
    page_end: int | None,
    bbox: Any,
    source_refs: list[str],
    source_block_ids: list[str],
    object_id: str | None = None,
    indexable: bool = True,
    retrieval_weight: float = 1.0,
    parent_chunk_id: str | None = None,
    flags: list[str] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_embedding = normalize_prose(embedding_text)
    identity = "|".join(
        [
            str(document["paper_version_id"]),
            content_type,
            content_role,
            *source_refs,
            sha256_text(normalized_embedding),
        ]
    )
    source_chunk_id = stable_uuid(identity)
    all_flags = sorted(
        set((flags or []) + quality_flags(normalized_embedding, require_text=indexable))
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "source_chunk_id": source_chunk_id,
        "ragflow_chunk_id": None,
        "paper_id": document["paper_id"],
        "paper_version_id": document["paper_version_id"],
        "dataset_id": None,
        "document_id": None,
        "content": normalized_embedding,
        "raw_content": raw_content,
        "embedding_text": normalized_embedding,
        "content_type": content_type,
        "content_role": content_role,
        "section": section_path[-1] if section_path else None,
        "section_path": section_path,
        "section_role": classify_section(section_path),
        "page_start": page_start,
        "page_end": page_end,
        "printed_page_label_start": None,
        "printed_page_label_end": None,
        "bbox": bbox,
        "source_ref": source_refs[0] if source_refs else None,
        "source_refs": source_refs,
        "source_block_ids": source_block_ids,
        "object_id": object_id,
        "parent_chunk_id": parent_chunk_id,
        "prev_chunk_id": None,
        "next_chunk_id": None,
        "indexable": indexable,
        "retrieval_weight": retrieval_weight,
        "parser_name": document.get("parser_name"),
        "parser_version": document.get("parser_version"),
        "cleaning_version": CLEANING_VERSION,
        "content_sha256": sha256_text(normalized_embedding),
        "quality_flags": all_flags,
        "provenance": provenance or {},
    }


def build_metadata_chunk(document: dict[str, Any], _blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a bibliographic-only chunk for title-oriented questions.

    Page-one paragraphs are deliberately excluded.  They often include part of
    the abstract, which made the metadata chunk a misleading candidate for
    research-problem queries.  Abstract and body chunks are generated
    separately by ``group_text_blocks`` and are the authoritative evidence for
    content questions.
    """
    title = str(document.get("title") or document.get("file_name") or "")
    file_name = str(document.get("file_name") or "")
    raw = "\n".join(part for part in (title, file_name) if part)
    embedding = (
        "文献元数据（仅用于题名、文件信息等书目查询）\n"
        f"论文标题：{title}\n"
        f"文件名：{file_name}"
    )
    return chunk_record(
        document=document,
        content_type="metadata",
        content_role="paper_metadata",
        section_path=[],
        raw_content=raw,
        embedding_text=embedding,
        page_start=1,
        page_end=1,
        bbox=None,
        source_refs=[f"paper://{document['paper_id']}/metadata"],
        source_block_ids=[],
        retrieval_weight=0.2,
        provenance={
            "scope": "bibliographic_only",
            "excluded_content": "page_one_paragraphs",
        },
    )


def is_spurious_numeric_heading(text: str) -> bool:
    """Identify MinerU's occasional table-cell sequence misclassified as a heading."""
    return bool(re.fullmatch(r"(?:\d+\s+){2,}\d+", normalize_prose(text)))


def html_to_prose(markup: str) -> str:
    """Recover readable prose from MinerU's HTML-wrapped algorithm body."""
    value = re.sub(r"</?(?:br|div|p|li|tr)[^>]*>", "\n", markup, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", "", value)
    return normalize_prose(html.unescape(value))


def repair_mineru_blocks(
    mineru_clean_dir: Path,
    blocks: list[dict[str, Any]],
    media_objects: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Apply narrow, auditable repairs for known MinerU structural mistakes.

    MinerU can emit an algorithm as a ``code_body`` while leaving its ordinary
    ``text`` field empty.  It can also mistake a table's repeated numeric cells
    for a section heading.  The raw export remains untouched; corrections are
    applied only to the in-memory input of the second-clean stage.
    """
    raw_files = sorted((mineru_clean_dir / "raw_mineru").rglob("*_content_list.json"))
    raw_items: list[dict[str, Any]] = []
    if raw_files:
        try:
            raw_content = read_json(raw_files[0])
            if isinstance(raw_content, list):
                raw_items = [item for item in raw_content if isinstance(item, dict)]
        except (OSError, json.JSONDecodeError):
            pass

    repaired: list[dict[str, Any]] = []
    repaired_paths: dict[str, list[str]] = {}
    repairs = Counter()
    last_valid_section_path: list[str] = []
    noisy_section_path: list[str] | None = None

    for original in blocks:
        block = {**original, "section_path": list(original.get("section_path") or [])}
        source_index = block.get("source_index")
        raw_item = (
            raw_items[source_index]
            if isinstance(source_index, int) and 0 <= source_index < len(raw_items)
            else {}
        )

        if block.get("content_role") == "section_heading":
            if is_spurious_numeric_heading(str(block.get("normalized_text") or "")):
                noisy_section_path = block["section_path"]
                block.update(
                    {
                        "content_type": "navigation",
                        "content_role": "page_footnote",
                        "indexable": False,
                        "repair_provenance": {"rule": "drop_spurious_numeric_heading"},
                    }
                )
                repairs["spurious_numeric_headings_removed"] += 1
            else:
                last_valid_section_path = block["section_path"]
                noisy_section_path = None
        elif noisy_section_path and block["section_path"] == noisy_section_path:
            block["section_path"] = list(last_valid_section_path)
            block["repair_provenance"] = {"rule": "restore_section_after_numeric_heading"}
            repairs["section_paths_restored"] += 1

        if (
            block.get("content_role") == "paragraph"
            and not str(block.get("normalized_text") or "").strip()
            and raw_item.get("type") == "code"
            and raw_item.get("code_body")
        ):
            recovered = html_to_prose(str(raw_item["code_body"]))
            if recovered:
                block.update(
                    {
                        "content_type": "text",
                        "content_role": "algorithm",
                        "raw_text": str(raw_item["code_body"]),
                        "normalized_text": recovered,
                        "quality_flags": quality_flags(recovered, require_text=True),
                        "repair_provenance": {"rule": "recover_algorithm_code_body"},
                    }
                )
                repairs["algorithm_blocks_recovered"] += 1

        repaired.append(block)
        repaired_paths[str(block["block_id"])] = block["section_path"]

    if not any(classify_section(block["section_path"]) == "abstract" for block in repaired):
        for block in repaired:
            text = str(block.get("normalized_text") or "")
            if (
                block.get("content_role") == "paragraph"
                and block.get("page_start") == 1
                and len(text) >= 280
            ):
                block["section_path"] = [*block["section_path"], "Abstract"]
                block["repair_provenance"] = {"rule": "infer_unlabelled_first_page_abstract"}
                repaired_paths[str(block["block_id"])] = block["section_path"]
                repairs["abstracts_inferred"] += 1
                break

    repaired_media = []
    for original in media_objects:
        media = {**original, "section_path": list(original.get("section_path") or [])}
        repaired_path = repaired_paths.get(str(media.get("block_id")))
        if repaired_path is not None and media["section_path"] != repaired_path:
            media["section_path"] = repaired_path
            media["repair_provenance"] = {"rule": "restore_section_after_numeric_heading"}
        repaired_media.append(media)
    return repaired, repaired_media, dict(repairs)


def split_text_at_token_limit(text: str, max_tokens: int) -> list[str]:
    """Split oversized prose while preserving all text and favoring natural breaks."""
    normalized = normalize_prose(text)
    if not normalized:
        return []
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")

    fragments: list[str] = []
    remaining = normalized
    while estimate_tokens(remaining) > max_tokens:
        low, high, best = 1, len(remaining), 0
        while low <= high:
            middle = (low + high) // 2
            if estimate_tokens(remaining[:middle]) <= max_tokens:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        # A single character can always fit when max_tokens is positive.
        best = max(1, best)
        minimum_preferred_cut = max(1, int(best * 0.55))
        preferred_cuts = [
            index + 1
            for index, char in enumerate(remaining[:best])
            if index + 1 >= minimum_preferred_cut
            and char in _PREFERRED_TRUNCATION_BOUNDARIES
        ]
        cut = preferred_cuts[-1] if preferred_cuts else best
        fragment = remaining[:cut].rstrip()
        if fragment:
            fragments.append(fragment)
        remaining = remaining[cut:].lstrip()
    if remaining:
        fragments.append(remaining)
    return fragments


def split_block_at_token_limit(block: dict[str, Any], max_tokens: int) -> list[dict[str, Any]]:
    """Expose oversized source blocks as mergeable fragments for chunking."""
    fragments = split_text_at_token_limit(str(block["normalized_text"]), max_tokens)
    if len(fragments) <= 1:
        return [block]
    count = len(fragments)
    return [
        {
            **block,
            "normalized_text": text,
            "chunk_fragment": {
                "block_id": block["block_id"],
                "fragment_index": index,
                "fragment_count": count,
            },
        }
        for index, text in enumerate(fragments, start=1)
    ]


def normalize_formula_spacing(equation: str) -> str:
    """Repair character-spaced MinerU math without changing ordinary formulas."""
    value = normalize_text(equation)
    command_space_marker = "\x00"
    value = re.sub(
        r"(\\[A-Za-z]+)\s+(?=[A-Za-z])", rf"\1{command_space_marker}", value
    )
    value = re.sub(r"(?<=[A-Za-z])\s+(?=[A-Za-z])", "", value)
    value = re.sub(r"(?<=[A-Za-z])\s+_\s*", "_", value)
    value = re.sub(r"(?<=[A-Za-z])\s+\\_\s*", r"\\_", value)
    value = re.sub(r"(?<=_)\s+(?=[A-Za-z{])", "", value)
    value = re.sub(r"(?<=\\_)\s+(?=[A-Za-z])", "", value)
    value = re.sub(r"(?<=\d)\.\s+(?=\d)", ".", value)
    value = re.sub(r"(?<=[{^])\s+(?=-\s*\d)", "", value)
    value = re.sub(r"(?<=-)\s+(?=\d)", "", value)
    return value.replace(command_space_marker, " ")


def group_text_blocks(
    document: dict[str, Any], blocks: list[dict[str, Any]], target_tokens: int, max_tokens: int
) -> list[dict[str, Any]]:
    grouped_by_section: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        if not block.get("indexable"):
            continue
        if block.get("content_role") != "paragraph":
            continue
        if not block.get("normalized_text"):
            continue
        grouped_by_section[tuple(block.get("section_path") or [])].append(block)

    chunks: list[dict[str, Any]] = []
    for section_key, section_blocks in grouped_by_section.items():
        section_path = list(section_key)
        parent_content = "\n\n".join(block["normalized_text"] for block in section_blocks)
        parent_id = stable_uuid(
            f"{document['paper_version_id']}:section-parent:{section_label(section_path)}"
        )
        parent = chunk_record(
            document=document,
            content_type="text",
            content_role="section_parent",
            section_path=section_path,
            raw_content=parent_content,
            embedding_text=f"章节：{section_label(section_path)}\n{parent_content}",
            page_start=min(block["page_start"] for block in section_blocks),
            page_end=max(block["page_end"] for block in section_blocks),
            bbox=None,
            source_refs=[block["source_ref"] for block in section_blocks],
            source_block_ids=[block["block_id"] for block in section_blocks],
            indexable=False,
            parent_chunk_id=None,
        )
        parent["source_chunk_id"] = parent_id
        chunks.append(parent)

        buffer: list[dict[str, Any]] = []
        buffer_tokens = 0
        groups: list[list[dict[str, Any]]] = []

        def flush() -> None:
            nonlocal buffer, buffer_tokens
            if not buffer:
                return
            groups.append(buffer)
            buffer = []
            buffer_tokens = 0

        for block in section_blocks:
            for fragment in split_block_at_token_limit(block, max_tokens):
                block_tokens = estimate_tokens(fragment["normalized_text"])
                if buffer and buffer_tokens + block_tokens > max_tokens:
                    flush()
                buffer.append(fragment)
                buffer_tokens += block_tokens
                if buffer_tokens >= target_tokens:
                    flush()
        flush()

        # A target-sized group can leave a very short final group.  Merge it
        # back when doing so stays within the hard limit.
        if len(groups) > 1:
            previous_text = "\n\n".join(block["normalized_text"] for block in groups[-2])
            final_text = "\n\n".join(block["normalized_text"] for block in groups[-1])
            if estimate_tokens(f"{previous_text}\n\n{final_text}") <= max_tokens:
                groups[-2].extend(groups[-1])
                groups.pop()

        for group in groups:
            text = "\n\n".join(block["normalized_text"] for block in group)
            role = "abstract" if classify_section(section_path) == "abstract" else "paragraph"
            fragment_provenance = [
                block["chunk_fragment"] for block in group if block.get("chunk_fragment")
            ]
            chunks.append(
                chunk_record(
                    document=document,
                    content_type="abstract" if role == "abstract" else "text",
                    content_role=role,
                    section_path=section_path,
                    raw_content=text,
                    embedding_text=f"章节：{section_label(section_path)}\n{text}",
                    page_start=min(block["page_start"] for block in group),
                    page_end=max(block["page_end"] for block in group),
                    bbox=[block.get("bbox") for block in group],
                    source_refs=[block["source_ref"] for block in group],
                    source_block_ids=[block["block_id"] for block in group],
                    parent_chunk_id=parent_id,
                    provenance=(
                        {
                            "truncation_merge": {
                                "strategy": "preferred_boundary_then_hard_cut",
                                "source_block_fragments": fragment_provenance,
                            }
                        }
                        if fragment_provenance
                        else {}
                    ),
                )
            )
    return chunks


def build_formula_chunks(
    document: dict[str, Any], blocks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        if block.get("content_role") != "display_formula":
            continue
        neighbours = [
            candidate
            for candidate in blocks[max(0, index - 2) : index + 3]
            if candidate.get("content_role") == "paragraph"
            and candidate.get("normalized_text")
        ]
        context = "\n".join(candidate["normalized_text"] for candidate in neighbours)
        original_equation = str(block["normalized_text"])
        equation = normalize_formula_spacing(original_equation)
        formula_flags = [
            flag
            for flag in (block.get("quality_flags") or [])
            if flag != "possible_spaced_ocr_text"
        ]
        formula_flags.extend(quality_flags(equation, require_text=True))
        embedding = (
            f"章节：{section_label(block.get('section_path') or [])}\n"
            f"公式：{equation}\n上下文：{context}"
        )
        chunks.append(
            chunk_record(
                document=document,
                content_type="formula",
                content_role="display_formula",
                section_path=block.get("section_path") or [],
                raw_content=equation,
                embedding_text=embedding,
                page_start=block.get("page_start"),
                page_end=block.get("page_end"),
                bbox=block.get("bbox"),
                source_refs=[block["source_ref"]],
                source_block_ids=[block["block_id"]],
                flags=formula_flags,
                provenance=(
                    {"mineru_repair": {"rule": "normalize_spaced_formula"}}
                    if equation != original_equation
                    else {}
                ),
            )
        )
    return chunks


def build_algorithm_chunks(
    document: dict[str, Any], blocks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for block in blocks:
        if block.get("content_role") != "algorithm" or not block.get("normalized_text"):
            continue
        text = str(block["normalized_text"])
        chunks.append(
            chunk_record(
                document=document,
                content_type="text",
                content_role="algorithm",
                section_path=block.get("section_path") or [],
                raw_content=str(block.get("raw_text") or text),
                embedding_text=(
                    f"Section: {section_label(block.get('section_path') or [])}\n"
                    f"Algorithm:\n{text}"
                ),
                page_start=block.get("page_start"),
                page_end=block.get("page_end"),
                bbox=block.get("bbox"),
                source_refs=[block["source_ref"]],
                source_block_ids=[block["block_id"]],
                flags=block.get("quality_flags") or [],
                provenance={"mineru_repair": block.get("repair_provenance") or {}},
            )
        )
    return chunks


def select_table_markdown(
    media: dict[str, Any], ocr: dict[str, Any] | None
) -> tuple[str, list[list[str]], str]:
    if ocr and ocr.get("status") == "success":
        candidates = ocr.get("table_markdown_candidates") or []
        matrices = ocr.get("table_matrices") or []
        if candidates:
            matrix = matrices[0] if matrices else []
            return str(candidates[0]), matrix, "baidu_table_v2"
        html_candidates = ocr.get("table_html_candidates") or []
        if html_candidates:
            matrix = html_table_to_matrix(str(html_candidates[0]))
            return matrix_to_markdown(matrix), matrix, "baidu_table_v2"
    mineru_matrix = html_table_to_matrix(str(media.get("table_html") or ""))
    if mineru_matrix:
        return matrix_to_markdown(mineru_matrix), mineru_matrix, "mineru_table_html"
    if ocr:
        candidates = ocr.get("table_markdown_candidates") or []
        if candidates:
            return str(candidates[0]), [], "baidu_table_v2"
        html_candidates = ocr.get("table_html_candidates") or []
        if html_candidates:
            matrix = html_table_to_matrix(str(html_candidates[0]))
            return matrix_to_markdown(matrix), matrix, "baidu_table_v2"
    return "", [], "none"


def build_media_chunks(
    document: dict[str, Any],
    media_objects: list[dict[str, Any]],
    ocr_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for media in media_objects:
        object_id = str(media["object_id"])
        ocr = ocr_by_id.get(object_id)
        caption = "\n".join(media.get("caption") or [])
        nearby = "\n".join(media.get("nearby_text") or [])
        section_path = media.get("section_path") or []
        object_type = str(media.get("object_type") or "image")
        source_ref = str(media.get("source_ref"))
        base_flags = list(media.get("quality_flags") or [])
        if ocr and ocr.get("status") == "failed":
            base_flags.append("ocr_failed")
        elif ocr and ocr.get("status") == "partial":
            base_flags.append("ocr_partial")

        if object_type == "table":
            markdown, matrix, table_source = select_table_markdown(media, ocr)
            raw = "\n".join(part for part in (caption, markdown) if part)
            overview = (
                f"章节：{section_label(section_path)}\n"
                f"表格标题：{caption}\n"
                f"表格内容：\n{markdown}\n"
                f"相关正文：{nearby}"
            )
            overview_chunk = chunk_record(
                document=document,
                content_type="table",
                content_role="table_overview",
                section_path=section_path,
                raw_content=raw,
                embedding_text=overview,
                page_start=media.get("page_start"),
                page_end=media.get("page_end"),
                bbox=media.get("pdf_bbox"),
                source_refs=[source_ref],
                source_block_ids=[media.get("block_id")],
                object_id=object_id,
                flags=base_flags + ([] if markdown else ["table_content_missing"]),
                provenance={"table_source": table_source},
            )
            chunks.append(overview_chunk)
            if matrix and len(matrix) > 3:
                header_rows = matrix[:2]
                for start in range(2, len(matrix), 4):
                    group = header_rows + matrix[start : start + 4]
                    group_markdown = matrix_to_markdown(group)
                    chunks.append(
                        chunk_record(
                            document=document,
                            content_type="table",
                            content_role="table_rows",
                            section_path=section_path,
                            raw_content=group_markdown,
                            embedding_text=(
                                f"章节：{section_label(section_path)}\n"
                                f"表格标题：{caption}\n表格行：\n{group_markdown}"
                            ),
                            page_start=media.get("page_start"),
                            page_end=media.get("page_end"),
                            bbox=media.get("pdf_bbox"),
                            source_refs=[f"{source_ref}/rows/{start}-{min(start + 3, len(matrix) - 1)}"],
                            source_block_ids=[media.get("block_id")],
                            object_id=object_id,
                            parent_chunk_id=overview_chunk["source_chunk_id"],
                            provenance={"table_source": table_source},
                        )
                    )
            continue

        ocr_text = str((ocr or {}).get("ocr_text") or "")
        descriptions = (ocr or {}).get("visual_descriptions") or []
        derived_description = "\n".join(str(value) for value in descriptions if value)
        raw_content = "\n".join(part for part in (caption, ocr_text, nearby) if part)
        embedding_parts = [
            f"章节：{section_label(section_path)}",
            f"图片标题：{caption}" if caption else "",
            f"图片内文字：{ocr_text}" if ocr_text else "",
            f"相关正文：{nearby}" if nearby else "",
        ]
        if derived_description:
            embedding_parts.append(f"模型派生图片说明：{derived_description}")
        chunks.append(
            chunk_record(
                document=document,
                content_type="figure" if object_type == "image" else "chart",
                content_role="figure_derived" if derived_description else "figure_source",
                section_path=section_path,
                raw_content=raw_content,
                embedding_text="\n".join(part for part in embedding_parts if part),
                page_start=media.get("page_start"),
                page_end=media.get("page_end"),
                bbox=media.get("pdf_bbox"),
                source_refs=[source_ref],
                source_block_ids=[media.get("block_id")],
                object_id=object_id,
                flags=base_flags,
                provenance={
                    "ocr_processors": (ocr or {}).get("processors") or [],
                    "derived_description": bool(derived_description),
                    "image_path": media.get("image_path"),
                },
            )
        )
    return chunks


def build_reference_chunks(
    document: dict[str, Any], blocks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for block in blocks:
        if block.get("content_role") != "reference_entry":
            continue
        text = block.get("normalized_text") or ""
        if not text:
            continue
        chunks.append(
            chunk_record(
                document=document,
                content_type="text",
                content_role="reference_entry",
                section_path=block.get("section_path") or ["References"],
                raw_content=text,
                embedding_text=f"参考文献：{text}",
                page_start=block.get("page_start"),
                page_end=block.get("page_end"),
                bbox=block.get("bbox"),
                source_refs=[block["source_ref"]],
                source_block_ids=[block["block_id"]],
                retrieval_weight=0.35,
            )
        )
    return chunks


def link_chunks(chunks: list[dict[str, Any]]) -> None:
    visible = [chunk for chunk in chunks if chunk.get("indexable")]
    visible.sort(
        key=lambda chunk: (
            chunk.get("page_start") or 0,
            chunk.get("content_role") or "",
            chunk["source_chunk_id"],
        )
    )
    for index, chunk in enumerate(visible):
        chunk["prev_chunk_id"] = visible[index - 1]["source_chunk_id"] if index else None
        chunk["next_chunk_id"] = (
            visible[index + 1]["source_chunk_id"] if index + 1 < len(visible) else None
        )


def ragflow_row(chunk: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        key: chunk.get(key)
        for key in (
            "paper_id",
            "paper_version_id",
            "source_chunk_id",
            "content_type",
            "content_role",
            "section",
            "section_path",
            "section_role",
            "page_start",
            "page_end",
            "source_ref",
            "object_id",
            "parent_chunk_id",
            "prev_chunk_id",
            "next_chunk_id",
            "retrieval_weight",
            "quality_flags",
            "parser_version",
            "cleaning_version",
        )
    }
    return {
        "document_id": chunk["source_chunk_id"],
        "content": chunk["embedding_text"],
        "metadata": metadata,
        "questions": [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mineru-clean-dir", type=Path, required=True)
    parser.add_argument("--ocr-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=350)
    parser.add_argument("--max-tokens", type=int, default=550)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.target_tokens < 1 or args.max_tokens < 1:
        raise ValueError("target_tokens and max_tokens must both be at least 1")
    if args.target_tokens > args.max_tokens:
        raise ValueError("target_tokens cannot exceed max_tokens")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    document = read_json(args.mineru_clean_dir / "document.json")
    blocks = list(iter_jsonl(args.mineru_clean_dir / "blocks.jsonl"))
    media_objects = list(iter_jsonl(args.mineru_clean_dir / "media_objects.jsonl"))
    blocks, media_objects, mineru_repairs = repair_mineru_blocks(
        args.mineru_clean_dir, blocks, media_objects
    )
    ocr_path = args.ocr_dir / "ocr_results.jsonl"
    ocr_by_id = (
        {row["object_id"]: row for row in iter_jsonl(ocr_path)}
        if ocr_path.exists()
        else {}
    )

    chunks = [build_metadata_chunk(document, blocks)]
    chunks.extend(
        group_text_blocks(
            document, blocks, target_tokens=args.target_tokens, max_tokens=args.max_tokens
        )
    )
    chunks.extend(build_formula_chunks(document, blocks))
    chunks.extend(build_algorithm_chunks(document, blocks))
    chunks.extend(build_media_chunks(document, media_objects, ocr_by_id))
    chunks.extend(build_reference_chunks(document, blocks))
    link_chunks(chunks)

    indexable = [chunk for chunk in chunks if chunk.get("indexable")]
    all_flags = Counter(flag for chunk in chunks for flag in chunk["quality_flags"])
    ocr_failures = sum(result.get("status") == "failed" for result in ocr_by_id.values())
    ocr_partials = sum(result.get("status") == "partial" for result in ocr_by_id.values())
    critical: list[str] = []
    if not document.get("title"):
        critical.append("missing_title")
    if len(indexable) < 5:
        critical.append("too_few_indexable_chunks")
    if not any(chunk["content_role"] == "abstract" for chunk in chunks):
        critical.append("missing_abstract_chunk")
    blocking_flags = {
        "unresolved_placeholder",
        "replacement_character",
        "missing_image",
        "table_content_missing",
        "ocr_failed",
    }
    has_blocking_flags = any(flag in blocking_flags for flag in all_flags)
    status = "FAILED" if critical else (
        "PARTIAL" if ocr_failures or ocr_partials or has_blocking_flags else "READY"
    )

    write_jsonl(args.output_dir / "chunks.jsonl", chunks)
    write_jsonl(
        args.output_dir / "ragflow_chunks.jsonl",
        (ragflow_row(chunk) for chunk in indexable),
    )
    manifest = {
        "schema_version": "paper_ragflow_manifest_v1",
        "paper_id": document["paper_id"],
        "paper_version_id": document["paper_version_id"],
        "document_name": document["file_name"],
        "title": document.get("title"),
        "file_sha256": document["file_sha256"],
        "chunk_count": len(indexable),
        "meta_fields": {
            "paper_id": document["paper_id"],
            "paper_version_id": document["paper_version_id"],
            "file_name": document["file_name"],
            "file_sha256": document["file_sha256"],
            "parser_version": document.get("parser_version"),
            "cleaning_version": CLEANING_VERSION,
            "schema_version": SCHEMA_VERSION,
        },
        "created_at": utc_now(),
    }
    write_json(args.output_dir / "document_manifest.json", manifest)
    quality_report = {
        "status": status,
        "critical_errors": critical,
        "paper_id": document["paper_id"],
        "paper_version_id": document["paper_version_id"],
        "total_chunks": len(chunks),
        "indexable_chunks": len(indexable),
        "parent_chunks": sum(not chunk.get("indexable") for chunk in chunks),
        "chunk_types": Counter(chunk["content_type"] for chunk in chunks),
        "chunk_roles": Counter(chunk["content_role"] for chunk in chunks),
        "quality_flags": all_flags,
        "media_objects": len(media_objects),
        "ocr_results": len(ocr_by_id),
        "ocr_failures": ocr_failures,
        "ocr_partials": ocr_partials,
        "mineru_repairs": mineru_repairs,
        "generated_at": utc_now(),
    }
    write_json(args.output_dir / "quality_report.json", quality_report)
    print(json.dumps(quality_report, ensure_ascii=False, default=dict), flush=True)
    return 1 if status == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
