"""Extract paragraphs and tables from a DOCX in document order."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _text(element: ET.Element) -> str:
    parts: list[str] = []
    for node in element.iter():
        if node.tag == f"{W}t" and node.text:
            parts.append(node.text)
        elif node.tag == f"{W}tab":
            parts.append("\t")
        elif node.tag == f"{W}br":
            parts.append("\n")
    return "".join(parts).strip()


def extract(path: Path) -> list[dict[str, object]]:
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    body = root.find(f"{W}body")
    if body is None:
        return []
    blocks: list[dict[str, object]] = []
    for child in body:
        if child.tag == f"{W}p":
            text = _text(child)
            if text:
                style_node = child.find(f"{W}pPr/{W}pStyle")
                style = style_node.get(f"{W}val") if style_node is not None else None
                blocks.append({"type": "paragraph", "style": style, "text": text})
        elif child.tag == f"{W}tbl":
            rows: list[list[str]] = []
            for row in child.findall(f"{W}tr"):
                rows.append([_text(cell) for cell in row.findall(f"{W}tc")])
            blocks.append({"type": "table", "rows": rows})
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    blocks = extract(args.source)
    content = json.dumps(blocks, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
