"""Export a PDF through the application's native PyMuPDF parser."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any


def serialize_block(block: Any) -> dict[str, Any]:
    return {
        "block_id": block.block_id,
        "page_number": block.page_number,
        "section_title": block.section_title,
        "content_type": block.content_type,
        "source_ref": block.source_ref,
        "content": block.content,
        "metadata": block.metadata,
    }


def serialize_media(item: Any) -> dict[str, Any]:
    data = asdict(item)
    return data


def markdown(document_name: str, blocks: list[dict[str, Any]], media: list[dict[str, Any]]) -> str:
    media_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in media:
        media_by_page[int(item["page_number"])].append(item)
    lines = [
        f"# {document_name}\n\n",
        "- Parser: PyMuPDF native_pdf\n",
        f"- Blocks: {len(blocks)}\n",
        f"- Media objects: {len(media)}\n\n",
    ]
    current_page: int | None = None
    for block in blocks:
        page = int(block["page_number"])
        if page != current_page:
            current_page = page
            lines.append(f"## Page {page}\n\n")
        role = str(block.get("metadata", {}).get("content_role") or "")
        text = str(block["content"]).strip()
        if role in {"title", "heading"}:
            lines.append(f"### {text}\n\n")
        else:
            lines.append(f"{text}\n\n")
    for page in sorted(media_by_page):
        lines.append(f"### Native media objects on page {page}\n\n")
        for item in media_by_page[page]:
            lines.append(f"> [{item['kind']}] `{item['object_id']}`\n\n")
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "backend"))
    from app.workers.ingestion import _parse_native_pdf

    pdf_path = args.pdf.resolve()
    parsed = _parse_native_pdf(pdf_path)
    name = args.name or pdf_path.stem
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    blocks = [serialize_block(block) for block in parsed.blocks]
    media = [serialize_media(item) for item in parsed.media]
    artifact = {
        "parser": "PyMuPDF native_pdf",
        "source_file": pdf_path.name,
        "source_path": str(pdf_path),
        "block_count": len(blocks),
        "media_count": len(media),
        "blocks": blocks,
        "media_objects": media,
    }
    json_path = destination / f"{name}.pymupdf.json"
    markdown_path = destination / f"{name}.pymupdf.md"
    json_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(markdown(name, blocks, media), encoding="utf-8")
    print(json.dumps({
        "json": str(json_path),
        "markdown": str(markdown_path),
        "block_count": len(blocks),
        "media_count": len(media),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
