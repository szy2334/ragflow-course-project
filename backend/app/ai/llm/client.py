"""Async OpenAI-compatible Chat Completions client."""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, SecretStr, ValidationError

from ..errors import ModelOutputInvalid, ModelTransportError
from ..schemas import ModelConfigSnapshot
from .base import ChatMessage, ModelCallMetrics, StructuredModelResult

TModel = TypeVar("TModel", bound=BaseModel)
RETRY_BACKOFF_SECONDS = 0.5


class OpenAICompatibleClient:
    """Calls an OpenAI-compatible endpoint without coupling to a vendor SDK."""

    def __init__(
        self,
        api_key: str | SecretStr,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        secret = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        if not secret.strip():
            raise ValueError("api_key cannot be blank")
        self._api_key = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
        self._transport = transport

    async def invoke_structured(
        self,
        messages: list[ChatMessage],
        output_model: type[TModel],
        config: ModelConfigSnapshot,
    ) -> StructuredModelResult[TModel]:
        started = time.perf_counter()
        retry_count = 0
        input_tokens = 0
        output_tokens = 0
        mode = config.structured_mode

        try:
            content, usage, retries = await self._complete(messages, output_model, config, mode)
            retry_count += retries
            input_tokens += usage.get("prompt_tokens", usage.get("input_tokens", 0))
            output_tokens += usage.get("completion_tokens", usage.get("output_tokens", 0))
        except _StructuredModeUnsupported:
            content, usage, retries = await self._complete(
                messages, output_model, config, "json_object"
            )
            retry_count += retries + 1
            input_tokens += usage.get("prompt_tokens", usage.get("input_tokens", 0))
            output_tokens += usage.get("completion_tokens", usage.get("output_tokens", 0))

        try:
            output = output_model.model_validate_json(
                self._json_payload(content), strict=True
            )
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            retry_count += 1
            repair_mode = "prompt_json" if mode == "prompt_json" else "json_object"
            repair_messages = self._repair_messages(
                content, exc, output_model, compact=repair_mode == "prompt_json"
            )
            try:
                repaired, usage, retries = await self._complete(
                    repair_messages, output_model, config, repair_mode
                )
                retry_count += retries
                input_tokens += usage.get("prompt_tokens", usage.get("input_tokens", 0))
                output_tokens += usage.get("completion_tokens", usage.get("output_tokens", 0))
                output = output_model.model_validate_json(
                    self._json_payload(repaired), strict=True
                )
            except (ValidationError, ValueError, json.JSONDecodeError) as repair_exc:
                raise ModelOutputInvalid(
                    "model output failed strict schema validation"
                ) from repair_exc

        metrics = ModelCallMetrics(
            latency_ms=int((time.perf_counter() - started) * 1000),
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            retry_count=retry_count,
            model=config.model,
            model_config_version=config.config_version,
        )
        return StructuredModelResult(output=output, metrics=metrics)

    async def invoke_structured_stream(
        self,
        messages: list[ChatMessage],
        output_model: type[TModel],
        config: ModelConfigSnapshot,
        on_content: Callable[[str], Awaitable[None]],
    ) -> StructuredModelResult[TModel]:
        """Stream model content while preserving the same final schema contract."""
        started = time.perf_counter()
        retry_count = 0
        input_tokens = 0
        output_tokens = 0
        mode = config.structured_mode

        try:
            content, usage, retries = await self._complete_stream(
                messages, output_model, config, mode, on_content
            )
            retry_count += retries
        except _StructuredModeUnsupported:
            content, usage, retries = await self._complete_stream(
                messages, output_model, config, "json_object", on_content
            )
            retry_count += retries + 1
        input_tokens += usage.get("prompt_tokens", usage.get("input_tokens", 0))
        output_tokens += usage.get("completion_tokens", usage.get("output_tokens", 0))

        try:
            output = output_model.model_validate_json(self._json_payload(content), strict=True)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            # A repair must be completed before it can be made visible as a final answer.
            retry_count += 1
            repair_mode = "prompt_json" if mode == "prompt_json" else "json_object"
            repair_messages = self._repair_messages(
                content, exc, output_model, compact=repair_mode == "prompt_json"
            )
            try:
                repaired, usage, retries = await self._complete(
                    repair_messages, output_model, config, repair_mode
                )
                retry_count += retries
                input_tokens += usage.get("prompt_tokens", usage.get("input_tokens", 0))
                output_tokens += usage.get("completion_tokens", usage.get("output_tokens", 0))
                output = output_model.model_validate_json(
                    self._json_payload(repaired), strict=True
                )
            except (ValidationError, ValueError, json.JSONDecodeError) as repair_exc:
                raise ModelOutputInvalid(
                    "model output failed strict schema validation"
                ) from repair_exc

        metrics = ModelCallMetrics(
            latency_ms=int((time.perf_counter() - started) * 1000),
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            retry_count=retry_count,
            model=config.model,
            model_config_version=config.config_version,
        )
        return StructuredModelResult(output=output, metrics=metrics)

    async def _complete(
        self,
        messages: list[ChatMessage],
        output_model: type[BaseModel],
        config: ModelConfigSnapshot,
        mode: str,
    ) -> tuple[str, dict[str, int], int]:
        payload_messages = self._messages_for_mode(messages, output_model, mode)
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in payload_messages
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_output_tokens,
        }
        response_format = self._response_format(mode, output_model)
        if response_format is not None:
            payload["response_format"] = response_format
        if config.reasoning_effort is not None:
            payload["reasoning_effort"] = config.reasoning_effort
        if config.enable_thinking is not None:
            payload["enable_thinking"] = config.enable_thinking
        headers = {
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        base_url = str(config.base_url).rstrip("/")
        endpoint = (
            f"{base_url}/chat/completions"
            if base_url.endswith("/v1")
            else f"{base_url}/v1/chat/completions"
        )
        retries = 0
        async with httpx.AsyncClient(
            transport=self._transport,
            timeout=config.timeout_seconds,
        ) as client:
            for attempt in range(2):
                try:
                    response = await client.post(endpoint, headers=headers, json=payload)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    if attempt == 0:
                        retries += 1
                        await asyncio.sleep(RETRY_BACKOFF_SECONDS)
                        continue
                    raise ModelTransportError("model endpoint unavailable") from exc

                if response.status_code in {400, 422} and mode == "json_schema":
                    raise _StructuredModeUnsupported
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt == 0:
                        retries += 1
                        await asyncio.sleep(RETRY_BACKOFF_SECONDS)
                        continue
                    raise ModelTransportError(
                        f"model endpoint returned retryable status {response.status_code}"
                    )
                try:
                    response.raise_for_status()
                    body = response.json()
                    content = body["choices"][0]["message"]["content"]
                except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                    raise ModelTransportError("invalid model endpoint response") from exc
                if isinstance(content, list):
                    content = "".join(
                        str(item.get("text", "")) for item in content if isinstance(item, dict)
                    )
                if not isinstance(content, str) or not content.strip():
                    raise ModelOutputInvalid("model returned empty content")
                usage = body.get("usage", {})
                return content, usage if isinstance(usage, dict) else {}, retries
        raise ModelTransportError("model endpoint request exhausted")

    async def _complete_stream(
        self,
        messages: list[ChatMessage],
        output_model: type[BaseModel],
        config: ModelConfigSnapshot,
        mode: str,
        on_content: Callable[[str], Awaitable[None]],
    ) -> tuple[str, dict[str, int], int]:
        payload_messages = self._messages_for_mode(messages, output_model, mode)
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in payload_messages
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_output_tokens,
            "stream": True,
        }
        response_format = self._response_format(mode, output_model)
        if response_format is not None:
            payload["response_format"] = response_format
        if config.reasoning_effort is not None:
            payload["reasoning_effort"] = config.reasoning_effort
        if config.enable_thinking is not None:
            payload["enable_thinking"] = config.enable_thinking
        headers = {
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        base_url = str(config.base_url).rstrip("/")
        endpoint = (
            f"{base_url}/chat/completions"
            if base_url.endswith("/v1")
            else f"{base_url}/v1/chat/completions"
        )
        retries = 0
        async with httpx.AsyncClient(
            transport=self._transport,
            timeout=config.timeout_seconds,
        ) as client:
            for attempt in range(2):
                chunks: list[str] = []
                usage: dict[str, int] = {}
                try:
                    async with client.stream(
                        "POST", endpoint, headers=headers, json=payload
                    ) as response:
                        if response.status_code in {400, 422} and mode == "json_schema":
                            raise _StructuredModeUnsupported
                        if response.status_code == 429 or response.status_code >= 500:
                            if attempt == 0:
                                retries += 1
                                await asyncio.sleep(RETRY_BACKOFF_SECONDS)
                                continue
                            raise ModelTransportError(
                                f"model endpoint returned retryable status {response.status_code}"
                            )
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            event = line[5:].strip()
                            if not event:
                                continue
                            if event == "[DONE]":
                                break
                            try:
                                body = json.loads(event)
                            except json.JSONDecodeError as exc:
                                raise ModelTransportError(
                                    "invalid model stream event"
                                ) from exc
                            event_usage = body.get("usage", {})
                            if isinstance(event_usage, dict):
                                usage = event_usage
                            choices = body.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {}).get("content")
                            if isinstance(delta, list):
                                delta = "".join(
                                    str(item.get("text", ""))
                                    for item in delta
                                    if isinstance(item, dict)
                                )
                            if isinstance(delta, str) and delta:
                                chunks.append(delta)
                                await on_content(delta)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    if attempt == 0 and not chunks:
                        retries += 1
                        await asyncio.sleep(RETRY_BACKOFF_SECONDS)
                        continue
                    raise ModelTransportError("model endpoint unavailable") from exc
                except httpx.HTTPError as exc:
                    raise ModelTransportError("invalid model endpoint response") from exc

                content = "".join(chunks)
                if not content.strip():
                    raise ModelOutputInvalid("model returned empty content")
                return content, usage, retries
        raise ModelTransportError("model endpoint request exhausted")

    @staticmethod
    def _response_format(mode: str, output_model: type[BaseModel]) -> dict[str, Any] | None:
        if mode == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": output_model.__name__[:64],
                    "strict": True,
                    "schema": output_model.model_json_schema(),
                },
            }
        if mode == "json_object":
            return {"type": "json_object"}
        return None

    @staticmethod
    def _messages_for_mode(
        messages: list[ChatMessage], output_model: type[BaseModel], mode: str
    ) -> list[ChatMessage]:
        if mode != "json_object":
            return messages
        schema = json.dumps(
            output_model.model_json_schema(), ensure_ascii=False, separators=(",", ":")
        )
        schema_instruction = (
            "Return exactly one JSON object that validates against this JSON Schema. "
            "Do not use Markdown fences or add explanatory text.\n"
            f"JSON Schema: {schema}"
        )
        if messages and messages[0].role == "system":
            # Some OpenAI-compatible providers, including SiliconFlow, reject a
            # second system message even when both are at the start.  Keep one
            # leading system message by merging our schema instruction with the
            # prompt's existing system rules.
            return [
                ChatMessage(
                    role="system",
                    content=f"{schema_instruction}\n\n{messages[0].content}",
                ),
                *messages[1:],
            ]
        return [ChatMessage(role="system", content=schema_instruction), *messages]

    @staticmethod
    def _json_payload(content: str) -> str:
        payload = content.strip()
        if not payload.startswith("```"):
            return payload
        _, separator, payload = payload.partition("\n")
        if not separator:
            return content.strip()
        payload = payload.strip()
        if payload.endswith("```"):
            payload = payload[:-3].rstrip()
        return payload

    @staticmethod
    def _repair_messages(
        previous_output: str,
        error: Exception,
        output_model: type[BaseModel],
        *,
        compact: bool = False,
    ) -> list[ChatMessage]:
        if compact:
            return [
                ChatMessage(
                    role="system",
                    content="Return one valid JSON object only. Do not use Markdown or add facts.",
                ),
                ChatMessage(
                    role="user",
                    content=(
                        "Repair the previous JSON with the required fields "
                        "for the requested result. "
                        f"Validation error: {error}. Previous output: {previous_output}"
                    ),
                ),
            ]
        return [
            ChatMessage(
                role="system",
                content="Repair JSON only. Do not add facts or explanations.",
            ),
            ChatMessage(
                role="user",
                content=(
                    "Return one JSON object matching this schema:\n"
                    f"{json.dumps(output_model.model_json_schema(), ensure_ascii=False)}\n"
                    f"Validation error: {error}\n"
                    f"Previous output: {previous_output}"
                ),
            ),
        ]


class _StructuredModeUnsupported(Exception):
    pass
