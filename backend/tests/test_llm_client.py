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
    assert bodies[0]["enable_thinking"] is False
    assert "secret" not in repr(client.__dict__)


@pytest.mark.asyncio
async def test_reasoning_effort_is_forwarded_when_configured(model_snapshot):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return response(valid_route_json())

    config = model_snapshot.model_copy(update={"reasoning_effort": "low"})
    client = OpenAICompatibleClient("secret", transport=httpx.MockTransport(handler))
    await client.invoke_structured(
        [ChatMessage(role="user", content="Return JSON")], RouteDecision, config
    )

    assert bodies[0]["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_max_tokens_is_omitted_when_output_limit_is_unset(model_snapshot):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return response(valid_route_json())

    config = model_snapshot.model_copy(update={"max_output_tokens": None})
    client = OpenAICompatibleClient("secret", transport=httpx.MockTransport(handler))
    await client.invoke_structured(
        [ChatMessage(role="user", content="Return JSON")], RouteDecision, config
    )

    assert "max_tokens" not in bodies[0]


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
async def test_prompt_json_omits_gateway_response_format_and_schema_injection(model_snapshot):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return response(valid_route_json())

    config = model_snapshot.model_copy(update={"structured_mode": "prompt_json"})
    client = OpenAICompatibleClient("secret", transport=httpx.MockTransport(handler))
    result = await client.invoke_structured(
        [ChatMessage(role="user", content="Return a compact JSON object")], RouteDecision, config
    )

    assert result.output.effective_route_type == "fact"
    assert "response_format" not in bodies[0]
    assert bodies[0]["messages"] == [{"role": "user", "content": "Return a compact JSON object"}]


@pytest.mark.asyncio
async def test_prompt_json_repair_includes_the_output_schema(model_snapshot):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return response("{}")
        return response(valid_route_json())

    config = model_snapshot.model_copy(update={"structured_mode": "prompt_json"})
    client = OpenAICompatibleClient("secret", transport=httpx.MockTransport(handler))
    result = await client.invoke_structured(
        [ChatMessage(role="user", content="Return JSON")], RouteDecision, config
    )

    assert result.metrics.retry_count == 1
    repair_prompt = bodies[1]["messages"][1]["content"]
    assert "JSON Schema:" in repair_prompt
    assert "initial_route_type" in repair_prompt
    assert "effective_route_type" in repair_prompt


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
async def test_markdown_fenced_json_is_accepted_without_a_repair(model_snapshot):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return response(f"```json\n{valid_route_json()}\n```")

    client = OpenAICompatibleClient("secret", transport=httpx.MockTransport(handler))
    result = await client.invoke_structured(
        [ChatMessage(role="user", content="Return JSON")], RouteDecision, model_snapshot
    )

    assert result.output.effective_route_type == "fact"
    assert calls == 1


def stream_response(content):
    payload = "\n\n".join(
        [
            "data: "
            + json.dumps({"choices": [{"delta": {"content": content}}]}),
            "data: [DONE]",
        ]
    )
    return httpx.Response(200, text=payload, headers={"content-type": "text/event-stream"})


@pytest.mark.asyncio
async def test_structured_stream_forwards_model_content(model_snapshot):
    bodies = []
    streamed: list[str] = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return stream_response(valid_route_json())

    async def collect(content: str) -> None:
        streamed.append(content)

    client = OpenAICompatibleClient("secret", transport=httpx.MockTransport(handler))
    result = await client.invoke_structured_stream(
        [ChatMessage(role="user", content="Return JSON")],
        RouteDecision,
        model_snapshot,
        collect,
    )

    assert result.output.effective_route_type == "fact"
    assert "".join(streamed) == valid_route_json()
    assert bodies[0]["stream"] is True
    assert bodies[0]["enable_thinking"] is False


@pytest.mark.asyncio
async def test_stream_omits_max_tokens_when_output_limit_is_unset(model_snapshot):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return stream_response(valid_route_json())

    async def ignore_delta(_content: str) -> None:
        pass

    config = model_snapshot.model_copy(update={"max_output_tokens": None})
    client = OpenAICompatibleClient("secret", transport=httpx.MockTransport(handler))
    await client.invoke_structured_stream(
        [ChatMessage(role="user", content="Return JSON")],
        RouteDecision,
        config,
        ignore_delta,
    )

    assert "max_tokens" not in bodies[0]


@pytest.mark.asyncio
async def test_thinking_parameter_is_omitted_when_not_configured(model_snapshot):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return response(valid_route_json())

    config = model_snapshot.model_copy(update={"enable_thinking": None})
    client = OpenAICompatibleClient("secret", transport=httpx.MockTransport(handler))
    await client.invoke_structured(
        [ChatMessage(role="user", content="Return JSON")], RouteDecision, config
    )

    assert "enable_thinking" not in bodies[0]


@pytest.mark.asyncio
async def test_json_object_merges_existing_system_message(model_snapshot):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return response(valid_route_json())

    config = model_snapshot.model_copy(update={"structured_mode": "json_object"})
    client = OpenAICompatibleClient("secret", transport=httpx.MockTransport(handler))
    await client.invoke_structured(
        [
            ChatMessage(role="system", content="Use only supplied evidence."),
            ChatMessage(role="user", content="Return JSON"),
        ],
        RouteDecision,
        config,
    )

    sent_messages = bodies[0]["messages"]
    assert [message["role"] for message in sent_messages] == ["system", "user"]
    assert "JSON Schema:" in sent_messages[0]["content"]
    assert "Use only supplied evidence." in sent_messages[0]["content"]


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
