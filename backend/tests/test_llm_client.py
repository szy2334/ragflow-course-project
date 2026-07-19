import json

import httpx
import pytest

from app.ai.errors import ModelOutputInvalid
from app.ai.llm import ChatMessage, OpenAICompatibleClient
from app.ai.schemas import RouteDecision


def valid_route_json():
    return json.dumps(
        {
            "initial_route_type": "fact",
            "effective_route_type": "fact",
            "standalone_question": "论文使用了什么数据集？",
            "review_dimensions": [],
            "needs_public_kb": False,
            "confidence": 0.9,
            "warnings": [],
        },
        ensure_ascii=False,
    )


def response(content, status=200):
    return httpx.Response(
        status,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


@pytest.mark.asyncio
async def test_json_schema_success(model_snapshot):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return response(valid_route_json())

    client = OpenAICompatibleClient("secret", transport=httpx.MockTransport(handler))
    result = await client.invoke_structured(
        [ChatMessage(role="user", content="Return JSON")], RouteDecision, model_snapshot
    )

    assert result.output.effective_route_type == "fact"
    assert bodies[0]["response_format"]["type"] == "json_schema"
    assert "secret" not in repr(client.__dict__)


@pytest.mark.asyncio
async def test_schema_mode_falls_back_to_json_object(model_snapshot):
    modes = []

    def handler(request):
        body = json.loads(request.content)
        modes.append(body["response_format"]["type"])
        if len(modes) == 1:
            return response("unsupported", status=400)
        return response(valid_route_json())

    client = OpenAICompatibleClient("secret", transport=httpx.MockTransport(handler))
    result = await client.invoke_structured(
        [ChatMessage(role="user", content="Return JSON")], RouteDecision, model_snapshot
    )

    assert result.output.initial_route_type == "fact"
    assert modes == ["json_schema", "json_object"]
    assert result.metrics.retry_count == 1


@pytest.mark.asyncio
async def test_429_is_retried_once(model_snapshot):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return response("busy", status=429)
        return response(valid_route_json())

    client = OpenAICompatibleClient("secret", transport=httpx.MockTransport(handler))
    result = await client.invoke_structured(
        [ChatMessage(role="user", content="Return JSON")], RouteDecision, model_snapshot
    )

    assert calls == 2
    assert result.metrics.retry_count == 1


@pytest.mark.asyncio
async def test_invalid_json_is_repaired_only_once(model_snapshot):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return response("not-json")

    client = OpenAICompatibleClient("secret", transport=httpx.MockTransport(handler))
    with pytest.raises(ModelOutputInvalid):
        await client.invoke_structured(
            [ChatMessage(role="user", content="Return JSON")], RouteDecision, model_snapshot
        )
    assert calls == 2


@pytest.mark.asyncio
async def test_base_url_already_ending_in_v1_is_not_duplicated(model_snapshot):
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return response(valid_route_json())

    config = model_snapshot.model_copy(update={"base_url": "https://models.example.test/v1"})
    client = OpenAICompatibleClient("secret", transport=httpx.MockTransport(handler))
    await client.invoke_structured(
        [ChatMessage(role="user", content="Return JSON")], RouteDecision, config
    )

    assert urls == ["https://models.example.test/v1/chat/completions"]
