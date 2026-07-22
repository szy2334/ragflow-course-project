from app.workers.ingestion import (
    BuiltChunk,
    _chunk_from_second_clean,
    _parse_native_pdf,
    _understanding_evidences,
)
from app.workers.second_clean_adapter import build_chunks as build_second_clean_chunks


def test_native_pdf_parser_preserves_text_and_image_geometry_without_ocr(tmp_path):
    import fitz

    path = tmp_path / "native.pdf"
    document = fitz.open()
    page = document.new_page(width=300, height=400)
    page.insert_text((40, 40), "Abstract", fontsize=12)
    page.insert_textbox(
        fitz.Rect(40, 55, 260, 120),
        "This paragraph comes from the PDF text layer and remains selectable.",
        fontsize=9,
    )
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 4, 4), False)
    pixmap.clear_with(255)
    page.insert_image(fitz.Rect(100, 150, 180, 210), pixmap=pixmap)
    document.save(path)
    document.close()

    parsed = _parse_native_pdf(path)

    assert any(block.section_title == "Abstract" for block in parsed.blocks)
    image = next(item for item in parsed.media if item.kind == "image")
    assert image.required is False
    assert image.image_url is None
    assert image.metadata["pdf_bbox"] == [100.0, 150.0, 180.0, 210.0]
    assert image.metadata["extraction_source"] == "native_pdf_image_object"


def test_upload_time_understanding_uses_local_bounded_evidence_only():
    chunks = [
        BuiltChunk(
            chunk_id=f"chunk-{index}",
            content=f"正文 {index}",
            content_type="text" if index else "reference",
            section_title="正文",
            page_number=index + 1,
            source_ref=f"page:{index + 1}",
            content_sha256="a" * 64,
            metadata={"indexable": index != 1},
        )
        for index in range(30)
    ]

    evidences = _understanding_evidences("paper-1", chunks)

    assert len(evidences) == 24
    assert all(item.document_id == "local:paper-1" for item in evidences)
    assert all(item.source_uri.startswith("paper://paper-1/") for item in evidences)
    assert all(item.content_type != "reference" for item in evidences)


def test_upload_time_understanding_normalizes_parser_specific_text_types():
    chunks = [
        BuiltChunk(
            chunk_id="abstract-1",
            content="This is the abstract.",
            content_type="abstract",
            section_title="Abstract",
            page_number=1,
            source_ref="page:1",
            content_sha256="a" * 64,
            metadata={"indexable": True},
        )
    ]

    evidences = _understanding_evidences("paper-1", chunks)

    assert [item.content_type for item in evidences] == ["text"]


def test_second_clean_chunks_are_local_and_keep_section_provenance():
    document = {
        "paper_id": "paper-1",
        "paper_version_id": "version-1",
        "title": "Evidence-aware reader",
        "file_name": "paper.pdf",
        "file_sha256": "a" * 64,
        "parser_name": "mineru",
        "parser_version": "mineru-v1",
    }
    blocks = []
    for index, section in enumerate(("Abstract", "Method", "Results", "Conclusion"), start=1):
        blocks.append(
            {
                "block_id": f"b-{index}",
                "raw_text": f"{section} " * 80,
                "normalized_text": f"{section} " * 80,
                "content_type": "text",
                "content_role": "paragraph",
                "section_path": [section],
                "page_start": index,
                "page_end": index,
                "bbox": None,
                "source_ref": f"page:{index}:block:1",
                "indexable": True,
                "quality_flags": [],
            }
        )

    cleaned = build_second_clean_chunks(
        document=document,
        blocks=blocks,
        media_objects=[
            {
                "object_id": "table-1",
                "object_type": "table",
                "caption": ["Table 1"],
                "nearby_text": [],
                "section_path": ["Results"],
                "page_start": 3,
                "page_end": 3,
                "pdf_bbox": None,
                "source_ref": "page:3:table:1",
                "block_id": "table-1",
                "quality_flags": [],
            }
        ],
        ocr_by_id={
            "table-1": {
                "status": "success",
                "ocr_text": "score | 0.91",
                "table_markdown_candidates": ["| score |\n| --- |\n| 0.91 |"],
            }
        },
    )
    rows = [_chunk_from_second_clean(chunk) for chunk in cleaned.chunks]

    assert cleaned.quality_report["knowledge_base_import"] == "not_required"
    assert cleaned.quality_report["status"] == "ready"
    assert any(chunk.metadata["content_role"] == "section_parent" for chunk in rows)
    assert any(chunk.metadata["section_path"] == ["Results"] for chunk in rows)
    assert any(chunk.metadata["content_role"] == "table_overview" for chunk in rows)
    assert all(chunk.metadata["cleaning_version"] == "paper_second_clean_v2" for chunk in rows)


def test_upload_time_understanding_samples_each_section_before_extra_chunks():
    chunks = [
        BuiltChunk(
            chunk_id=f"{section}-{index}",
            content=f"{section} evidence {index}",
            content_type="text",
            section_title=section,
            page_number=position * 10 + index,
            source_ref=f"page:{position * 10 + index}",
            content_sha256="a" * 64,
            metadata={"indexable": True, "section_path": [section]},
        )
        for position, section in enumerate(("Abstract", "Method", "Results", "Conclusion"), start=1)
        for index in range(1, 11)
    ]

    evidences = _understanding_evidences("paper-1", chunks)
    sampled_sections = {item.section_title for item in evidences}

    assert len(evidences) == 24
    assert sampled_sections == {"Abstract", "Method", "Results", "Conclusion"}
