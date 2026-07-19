"""Async OpenAI-compatible Chat Completions client."""

import asyncio
import json
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, SecretStr, ValidationError

from ..errors import ModelOutputInvalid, ModelTransportError
from ..schemas import ModelConfigSnapshot
from .base import ChatMessage, ModelCallMetrics, StructuredModelResult

TModel = TypeVar("TModel", bound=BaseModel)


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
            output = output_model.model_validate_json(content, strict=True)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            retry_count += 1
            repair_messages = self._repair_messages(content, exc, output_model)
            try:
                repaired, usage, retries = await self._complete(
                    repair_messages, output_model, config, "json_object"
                )
                retry_count += retries
                input_tokens += usage.get("prompt_tokens", usage.get("input_tokens", 0))
                output_tokens += usage.get("completion_tokens", usage.get("output_tokens", 0))
                output = output_model.model_validate_json(repaired, strict=True)
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
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_output_tokens,
            "response_format": self._response_format(mode, output_model),
        }
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
                        await asyncio.sleep(0)
                        continue
                    raise ModelTransportError("model endpoint unavailable") from exc

                if response.status_code in {400, 422} and mode == "json_schema":
                    raise _StructuredModeUnsupported
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt == 0:
                        retries += 1
                        await asyncio.sleep(0)
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
                        str(item.get("text", ""))
                        for item in content
                        if isinstance(item, dict)
                    )
                if not isinstance(content, str) or not content.strip():
                    raise ModelOutputInvalid("model returned empty content")
                usage = body.get("usage", {})
                return content, usage if isinstance(usage, dict) else {}, retries
        raise ModelTransportError("model endpoint request exhausted")

    @staticmethod
    def _response_format(mode: str, output_model: type[BaseModel]) -> dict[str, Any]:
        if mode == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": output_model.__name__[:64],
                    "strict": True,
                    "schema": output_model.model_json_schema(),
                },
            }
        return {"type": "json_object"}

    @staticmethod
    def _repair_messages(
        previous_output: str,
        error: Exception,
        output_model: type[BaseModel],
    ) -> list[ChatMessage]:
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
