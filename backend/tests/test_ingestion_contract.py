import pytest
from conftest import ScriptedLlm

from app.ai.agents import PaperUnderstandingAgent
from app.ai.prompts import PromptRepository
from app.ai.runner import AgentRunner
from app.ai.schemas import ConfigurationSnapshot, EvidenceItem, PaperSummary
from app.workers.ingestion import (
    BuiltChunk,
    _chunk_from_second_clean,
    _understanding_evidences,
)
from app.workers.second_clean_adapter import build_chunks as build_second_clean_chunks


@pytest.mark.asyncio
async def test_upload_time_summary_uses_shared_agent_runner(model_snapshot):
    llm = ScriptedLlm([PaperSummary(summary_markdown="# Summary\nSupported overview." )])
    agent = PaperUnderstandingAgent(AgentRunner(llm, PromptRepository()))
    evidence = EvidenceItem(
        evidence_id="U1",
        source_type="paper",
        paper_id="paper-1",
        document_id="local:paper-1",
        chunk_id="chunk-1",
        content_type="text",
        quote="The paper studies robust retrieval.",
        source_uri="paper://paper-1/chunk-1",
        retrieval_score=1.0,
    )

    summary, result = await agent.run_summary(
        evidences=[evidence],
        configuration=ConfigurationSnapshot(
            graph_version="v3",
            prompt_version="v1",
            schema_version="v1",
            model=model_snapshot,
        ),
    )

    assert summary.summary_markdown.startswith("# Summary")
    assert result.agent_name == "paper_understanding"
    assert llm.calls == ["PaperSummary"]


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
