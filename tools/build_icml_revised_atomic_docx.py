"""Create the approval-candidate ICML rule DOCX with atomic rule bullets."""

from __future__ import annotations

import argparse
import copy
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from build_icml_atomic_rules import (
    ATOMIC_SPLITS,
    REMOVED_RULES,
    REMOVED_SECTIONS,
    RULE_REPHRASES,
    SECTION_TITLE_REPHRASES,
)

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
XML = "{http://www.w3.org/XML/1998/namespace}"


def _paragraph_text(paragraph: ET.Element) -> str:
    return "".join(node.text or "" for node in paragraph.iter(f"{W}t")).strip()


def _replace_text(paragraph: ET.Element, value: str) -> None:
    nodes = list(paragraph.iter(f"{W}t"))
    if not nodes:
        run = ET.SubElement(paragraph, f"{W}r")
        nodes = [ET.SubElement(run, f"{W}t")]
    nodes[0].text = value
    nodes[0].set(f"{XML}space", "preserve")
    for node in nodes[1:]:
        node.text = ""


def build(source: Path, output: Path) -> dict[str, object]:
    with zipfile.ZipFile(source) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    root = ET.fromstring(files["word/document.xml"])
    body = root.find(f"{W}body")
    if body is None:
        raise RuntimeError("DOCX document body is missing")

    removed_rules: list[str] = []
    removed_section_content: list[str] = []
    split_sources: dict[str, list[str]] = {}
    rephrased: list[dict[str, object]] = []
    section = 0
    for paragraph in list(body.findall(f"{W}p")):
        text = _paragraph_text(paragraph)
        heading = re.fullmatch(r"(\d+)\.\s+(.+)", text)
        if heading:
            section = int(heading.group(1))
            if section in REMOVED_SECTIONS:
                body.remove(paragraph)
                removed_section_content.append(text)
                continue
            replacement_heading = SECTION_TITLE_REPHRASES.get(text)
            if replacement_heading:
                _replace_text(paragraph, replacement_heading)
                rephrased.append({"section": section, "original": text, "rephrased": replacement_heading})
            continue
        if section in REMOVED_SECTIONS:
            body.remove(paragraph)
            removed_section_content.append(text)
            continue
        if text in REMOVED_RULES:
            body.remove(paragraph)
            removed_rules.append(text)
            continue
        replacements = ATOMIC_SPLITS.get(text)
        if replacements:
            position = list(body).index(paragraph)
            body.remove(paragraph)
            for offset, replacement in enumerate(replacements):
                clone = copy.deepcopy(paragraph)
                _replace_text(clone, replacement)
                body.insert(position + offset, clone)
            split_sources[text] = replacements
            continue
        replacement = RULE_REPHRASES.get((section, text))
        if replacement:
            _replace_text(paragraph, replacement)
            rephrased.append({"section": section, "original": text, "rephrased": replacement})

    if set(removed_rules) != REMOVED_RULES:
        raise RuntimeError(f"Not all requested rules were removed: {sorted(REMOVED_RULES - set(removed_rules))}")
    removed_section_headings = {
        int(match.group(1))
        for text in removed_section_content
        if (match := re.fullmatch(r"(\d+)\.\s+(.+)", text))
    }
    if removed_section_headings != REMOVED_SECTIONS:
        raise RuntimeError(
            f"Not all requested sections were removed: {sorted(REMOVED_SECTIONS - removed_section_headings)}"
        )
    if set(split_sources) != set(ATOMIC_SPLITS):
        raise RuntimeError(f"Not all compound rules were split: {sorted(set(ATOMIC_SPLITS) - set(split_sources))}")

    files["word/document.xml"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return {
        "source": str(source),
        "output": str(output),
        "removed_rules": removed_rules,
        "removed_sections": sorted(REMOVED_SECTIONS),
        "removed_section_content": removed_section_content,
        "compound_rules_split": len(split_sources),
        "atomic_rules_from_compounds": sum(len(values) for values in split_sources.values()),
        "rephrased_rules_and_headings": len(rephrased),
        "rephrases": rephrased,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import json

    print(json.dumps(build(args.source, args.output), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
