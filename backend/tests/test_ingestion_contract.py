from app.workers.ingestion import (
    BuiltChunk,
    MediaObject,
    ParsedBlock,
    ParsedPaper,
    _build_chunks,
    _quality_report,
)


def test_structured_chunks_keep_roles_links_sources_and_parentage():
    parsed = ParsedPaper(
        blocks=[
            ParsedBlock("b1", "Main finding", 1, "Results", source_ref="page:1:block:1"),
            ParsedBlock(
                "b2",
                "Reference entry",
                2,
                "References",
                content_type="reference",
                source_ref="page:2:block:2",
            ),
        ],
        media=[
            MediaObject(
                "figure-1",
                "figure",
                3,
                "page:3:media:1",
                "object://figure-1",
                "Figure 1",
            )
        ],
    )

    chunks = _build_chunks("paper-1", parsed, {"figure-1": "axis labels"})
    by_id = {chunk.chunk_id: chunk for chunk in chunks}

    first = by_id["paper-1:text:1"]
    second = by_id["paper-1:text:2"]
    figure = by_id["paper-1:figure:figure-1"]
    figure_ocr = by_id["paper-1:figure:figure-1:ocr"]
    assert first.metadata["content_role"] == "paragraph"
    assert first.metadata["next_chunk_id"] == second.chunk_id
    assert second.metadata["prev_chunk_id"] == first.chunk_id
    assert second.metadata["content_role"] == "reference_entry"
    assert second.metadata["retrieval_weight"] < 1
    assert figure.metadata["content_role"] == "figure_overview"
    assert figure_ocr.metadata["content_role"] == "figure_ocr"
    assert figure_ocr.parent_chunk_id == figure.chunk_id
    assert not _quality_report(parsed, chunks, {"figure-1": "axis labels"})


def test_quality_gate_blocks_required_ocr_and_orphan_chunks():
    required_media = MediaObject(
        "table-1",
        "table",
        2,
        "page:2:media:1",
        "object://table-1",
        "Table 1",
    )
    parsed = ParsedPaper(
        blocks=[ParsedBlock("b1", "Text", 1, "Body", source_ref="page:1:block:1")],
        media=[required_media],
    )
    chunks = _build_chunks("paper-1", parsed, {})
    assert "ocr_missing:table-1" in _quality_report(parsed, chunks, {})

    orphan = BuiltChunk(
        chunk_id="orphan",
        content="row",
        content_type="table",
        section_title="Table",
        page_number=2,
        source_ref="page:2:media:1",
        content_sha256="a" * 64,
        object_id="table-1",
        parent_chunk_id="missing-parent",
    )
    assert "orphan_chunk:orphan" in _quality_report(parsed, [orphan], {"table-1": "row"})


def test_optional_media_without_ocr_does_not_create_an_empty_child():
    optional_media = MediaObject(
        "decoration-1",
        "figure",
        1,
        "page:1:media:1",
        None,
        "Decorative logo",
        required=False,
    )
    parsed = ParsedPaper(blocks=[], media=[optional_media])
    chunks = _build_chunks("paper-1", parsed, {})

    assert [chunk.chunk_id for chunk in chunks] == ["paper-1:figure:decoration-1"]
    assert not _quality_report(parsed, chunks, {})
