"""Export auditable MinerU + PyMuPDF fusion artifacts from local uploads."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import fitz


SPACE = re.compile(r"\s+")
MARKUP = re.compile(r"\$\^\{?[^$\s}]+\}?\$|[*_`#]")


def normalize_text(value: str) -> str:
    value = MARKUP.sub("", value or "")
    return SPACE.sub("", value).replace("-", "").lower()


def native_lines(pdf_path: Path) -> tuple[list[dict[str, Any]], dict[int, tuple[float, float]]]:
    lines: list[dict[str, Any]] = []
    sizes: dict[int, tuple[float, float]] = {}
    document = fitz.open(pdf_path)
    try:
        for page_number, page in enumerate(document, start=1):
            sizes[page_number] = (float(page.rect.width), float(page.rect.height))
            for block in page.get_text("rawdict").get("blocks", []):
                for line in block.get("lines", []):
                    line_spans: list[dict[str, Any]] = []
                    for raw_span in line.get("spans", []):
                        text = "".join(
                            str(character.get("c") or "")
                            for character in raw_span.get("chars", [])
                            if isinstance(character, dict)
                        ).strip()
                        if not text:
                            continue
                        font = str(raw_span.get("font") or "") or None
                        line_spans.append(
                            {
                                "page_number": page_number,
                                "text": text,
                                "normalized_text": normalize_text(text),
                                "bbox_pdf_pt": [float(item) for item in raw_span["bbox"]],
                                "font_name": re.sub(r"^[A-Z]{6}\+", "", font) if font else None,
                                "font_size_pt": round(float(raw_span.get("size") or 0), 3) or None,
                                "font_flags": int(raw_span.get("flags") or 0),
                                "color": int(raw_span.get("color") or 0),
                            }
                        )
                    if not line_spans:
                        continue
                    x0 = min(span["bbox_pdf_pt"][0] for span in line_spans)
                    y0 = min(span["bbox_pdf_pt"][1] for span in line_spans)
                    x1 = max(span["bbox_pdf_pt"][2] for span in line_spans)
                    y1 = max(span["bbox_pdf_pt"][3] for span in line_spans)
                    lines.append(
                        {
                            "page_number": page_number,
                            "text": "".join(span["text"] for span in line_spans),
                            "normalized_text": normalize_text(
                                "".join(span["text"] for span in line_spans)
                            ),
                            "bbox_pdf_pt": [x0, y0, x1, y1],
                            "spans": line_spans,
                        }
                    )
    finally:
        document.close()
    return lines, sizes


def match_spans(block: dict[str, Any], lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text = normalize_text(str(block.get("text") or ""))
    if len(text) < 3:
        return []
    page_number = int(block.get("page_idx") or 0) + 1
    candidates = [line for line in lines if line["page_number"] == page_number]
    best: tuple[float, list[dict[str, Any]]] | None = None
    prefix = text[: min(12, len(text))]
    start_indices = [
        index
        for index, line in enumerate(candidates)
        if len(line["normalized_text"]) >= 8
        and (
            prefix[:8] in line["normalized_text"]
            or line["normalized_text"][:8] in text
        )
    ]
    if not start_indices:
        return []
    # MinerU blocks may wrap across multiple native PDF lines. Score only
    # contiguous line ranges so repeated terms elsewhere on a page cannot join
    # an otherwise valid match.
    for start in start_indices:
        joined = ""
        for end in range(start, min(start + 60, len(candidates))):
            joined += candidates[end]["normalized_text"]
            if len(joined) < max(3, len(text) // 5):
                continue
            if len(joined) > len(text) * 1.25 + 12:
                break
            if text not in joined and joined not in text:
                continue
            score = -abs(len(text) - len(joined))
            if best is None or score > best[0]:
                best = (score, candidates[start : end + 1])
    if best is None:
        return []
    return [span for line in best[1] for span in line["spans"]]


def aggregate_style(spans: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not spans:
        return None
    font_counts = Counter(span["font_name"] for span in spans if span["font_name"])
    size_counts = Counter(span["font_size_pt"] for span in spans if span["font_size_pt"])
    flag_counts = Counter(span["font_flags"] for span in spans)
    x0 = min(span["bbox_pdf_pt"][0] for span in spans)
    y0 = min(span["bbox_pdf_pt"][1] for span in spans)
    x1 = max(span["bbox_pdf_pt"][2] for span in spans)
    y1 = max(span["bbox_pdf_pt"][3] for span in spans)
    return {
        "matched_span_count": len(spans),
        "font_name": font_counts.most_common(1)[0][0] if font_counts else None,
        "font_size_pt": size_counts.most_common(1)[0][0] if size_counts else None,
        "font_flags": flag_counts.most_common(1)[0][0] if flag_counts else None,
        "bbox_pdf_pt": [round(value, 3) for value in (x0, y0, x1, y1)],
    }


def markdown_for(block: dict[str, Any], style: dict[str, Any] | None) -> str:
    content_type = str(block.get("type") or "text")
    text = str(block.get("text") or "").strip()
    if content_type == "text" and text:
        level = block.get("heading_level", block.get("text_level"))
        try:
            level = int(level)
        except (TypeError, ValueError):
            level = 0
        if 1 <= level <= 6:
            return f"{'#' * level} {text}\n"
        return f"{text}\n"
    if content_type in {"image", "table"}:
        caption = text or f"{content_type.title()} detected by MinerU"
        return f"> [{content_type}] {caption}\n"
    return f"> [{content_type}] {text}\n" if text else ""


def export_one(name: str, pdf_path: Path, mineru_dir: Path, output_root: Path) -> dict[str, Any]:
    content_path = next(mineru_dir.glob("*_content_list.json"))
    blocks = json.loads(content_path.read_text(encoding="utf-8"))
    layout = json.loads((mineru_dir / "layout.json").read_text(encoding="utf-8"))
    lines, page_sizes = native_lines(pdf_path)
    fused_blocks: list[dict[str, Any]] = []
    matched = 0
    for index, block in enumerate(blocks, start=1):
        matches = match_spans(block, lines)
        style = aggregate_style(matches)
        if style:
            matched += 1
        page_number = int(block.get("page_idx") or 0) + 1
        fused_blocks.append(
            {
                "block_id": f"mineru-{index}",
                "page_number": page_number,
                "content_type": block.get("type"),
                "heading_level": block.get("text_level"),
                "text": block.get("text"),
                "mineru_bbox": block.get("bbox"),
                "mineru_source": "content_list.json",
                "native_pdf": style,
                "fusion_status": "matched" if style else "unmatched",
            }
        )
    destination = output_root / name
    destination.mkdir(parents=True, exist_ok=True)
    source_images = mineru_dir / "images"
    if source_images.exists():
        shutil.copytree(source_images, destination / "images", dirs_exist_ok=True)
    artifact = {
        "schema_version": "mineru-pymupdf-fusion-v1",
        "document": {
            "name": name,
            "source_pdf": str(pdf_path.resolve()),
            "mineru_export": str(mineru_dir.resolve()),
            "page_count": len(page_sizes),
            "page_sizes_pdf_pt": {str(page): [width, height] for page, (width, height) in page_sizes.items()},
            "mineru_layout_metadata": {
                key: layout.get(key)
                for key in ("_backend", "_effort", "_ocr_enable", "_version_name")
            },
        },
        "summary": {
            "mineru_block_count": len(fused_blocks),
            "native_pdf_line_count": len(lines),
            "matched_block_count": matched,
            "unmatched_block_count": len(fused_blocks) - matched,
            "matching_method": "same-page normalized text; contiguous native-line match",
        },
        "blocks": fused_blocks,
    }
    (destination / f"{name}.fused.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        f"# {name}: MinerU + PyMuPDF Fusion Result\n",
        f"- Source PDF: `{pdf_path.name}`\n",
        f"- MinerU blocks: {len(fused_blocks)}\n",
        f"- PyMuPDF native lines: {len(lines)}\n",
        f"- Matched blocks: {matched}/{len(fused_blocks)}\n",
        "- Each block's native font, size, flags, and PDF-point bbox are in the accompanying JSON.\n\n",
        "---\n\n",
    ]
    for fused in fused_blocks:
        lines.append(markdown_for(fused, fused["native_pdf"]))
        lines.append("\n")
    (destination / f"{name}.fused.md").write_text("".join(lines), encoding="utf-8")
    return artifact["summary"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    pairs = [
        (
            "BLIP",
            root / "backend/var/uploads/a376c6c3-a8fa-4769-b6a1-267c6725b4f5/4444a9d6-c865-4250-b67c-106f335a1048.pdf",
            root / "backend/var/uploads/db4664ab-bb8d-4716-acca-9fc7df06cb41/3eb60905-457d-4318-bf97-4c06103b33d1/mineru/raw_mineru/extracted",
        ),
        (
            "UMAP",
            root / "backend/var/uploads/fb0a22ec-7d5a-494b-93b0-2fffd4ee29d6/4f004c89-ec24-4b81-81c4-ebb7e923debd.pdf",
            root / "backend/var/uploads/8d468563-fb8c-464c-874a-5cf55760ba7a/df3be51d-18e0-4319-b77e-349f7ff77339/mineru/raw_mineru/extracted",
        ),
    ]
    summaries = {
        name: export_one(name, pdf_path, mineru_dir, args.output_dir)
        for name, pdf_path, mineru_dir in pairs
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
