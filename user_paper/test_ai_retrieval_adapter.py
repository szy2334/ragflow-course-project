import json
from pathlib import Path

import httpx
import pytest

from app.ai.schemas import RetrieveEvidenceRequest, RetrieveStandardsRequest
from user_paper.ai_retrieval_adapter import (
    AsyncRagflowRetrievalClient,
    JsonlPaperRegistry,
    RagflowRetrievalAdapter,
    RagflowRetrievalSettings,
)


def write_mapping(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "user_paper_ragflow_mapping_v2",
                "user_id": "user-1",
                "source_chunk_id": "source-1",
                "ragflow_chunk_id": "rag-1",
                "paper_id": "paper-1",
                "dataset_id": "dataset-1",
                "document_id": "document-1",
                "content_type": "chart",
                "content_role": "result_figure",
                "section": "Results",
                "page_start": 7,
                "source_ref": "page:7",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def request() -> RetrieveEvidenceRequest:
    return RetrieveEvidenceRequest(
        task_id="task-1",
        user_id="user-1",
        paper_ids=["paper-1"],
        standalone_question="图 2 展示了什么？",
        route_type="explain",
    )


@pytest.mark.asyncio
async def test_adapter_maps_authorised_ragflow_chunk_to_ai_evidence(tmp_path: Path) -> None:
    mapping = tmp_path / "chunk_mapping.jsonl"
    write_mapping(mapping)
    received: list[httpx.Request] = []

    async def handler(http_request: httpx.Request) -> httpx.Response:
        received.append(http_request)
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "chunks": [
                        {
                            "id": "rag-1",
                            "content": "图 2 比较了二维嵌入的局部结构。",
                            "similarity": 0.91,
                        }
                    ]
                },
            },
        )

    settings = RagflowRetrievalSettings(
        base_url="https://ragflow.example/api/v1",
        api_key="test-key",
    )
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = RagflowRetrievalAdapter(
        JsonlPaperRegistry([mapping]),
        AsyncRagflowRetrievalClient(settings, client=http_client),
    )

    evidence = await adapter.retrieve_paper(request())

    assert len(received) == 1
    assert received[0].url.path == "/api/v1/retrieval"
    assert evidence.query == "图 2 展示了什么？"
    assert evidence.items[0].evidence_id == "P1"
    assert evidence.items[0].content_type == "figure"
    assert evidence.items[0].paper_id == "paper-1"
    assert evidence.items[0].page_number == 7
    assert evidence.items[0].retrieval_score == 0.91
    await http_client.aclose()


@pytest.mark.asyncio
async def test_adapter_does_not_query_when_user_has_no_authorised_paper(tmp_path: Path) -> None:
    mapping = tmp_path / "chunk_mapping.jsonl"
    write_mapping(mapping)
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: pytest.fail("RAGFlow must not be called"))
    )
    settings = RagflowRetrievalSettings(
        base_url="https://ragflow.example/api/v1",
        api_key="test-key",
    )
    adapter = RagflowRetrievalAdapter(
        JsonlPaperRegistry([mapping]),
        AsyncRagflowRetrievalClient(settings, client=http_client),
    )
    unauthorised = request().model_copy(update={"user_id": "user-2"})

    evidence = await adapter.retrieve_paper(unauthorised)

    assert evidence.items == []
    assert evidence.warnings == ["NO_AUTHORISED_PAPER_TARGETS"]
    await http_client.aclose()


@pytest.mark.asyncio
async def test_standard_retrieval_is_explicitly_unconfigured(tmp_path: Path) -> None:
    mapping = tmp_path / "chunk_mapping.jsonl"
    write_mapping(mapping)
    settings = RagflowRetrievalSettings(
        base_url="https://ragflow.example/api/v1",
        api_key="test-key",
    )
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: pytest.fail()))
    adapter = RagflowRetrievalAdapter(
        JsonlPaperRegistry([mapping]),
        AsyncRagflowRetrievalClient(settings, client=http_client),
    )

    evidence = await adapter.retrieve_standards(
        RetrieveStandardsRequest(
            task_id="task-1",
            standalone_question="评价实验设计",
            route_type="review",
            dimensions=["experiment"],
        )
    )

    assert evidence.items == []
    assert evidence.warnings == ["PUBLIC_STANDARD_RETRIEVAL_NOT_CONFIGURED"]
    await http_client.aclose()
