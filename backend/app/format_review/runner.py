"""The mechanical LLM call boundary used by the format-review graph."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, TypeVar

from pydantic import BaseModel

from app.ai.llm import OpenAICompatibleClient
from app.ai.prompts.repository import PromptRepository
from app.ai.schemas import ConfigurationSnapshot
from app.core.config import Settings
from app.runtime.executor import snapshot_from_settings

TModel = TypeVar("TModel", bound=BaseModel)

_OUTPUT_TOKEN_BUDGETS = {
    "format_check": 800,
    "format_reflect": 240,
    "format_synthesis": 600,
}


class AgentRunner:
    """Renders a versioned prompt, validates output and returns raw metrics.

    Business nodes decide what is checked and how evidence is interpreted; this
    class intentionally owns only the provider call lifecycle.
    """

    def __init__(self, settings: Settings, configuration: dict[str, Any] | None) -> None:
        self._settings = settings
        self._prompts = PromptRepository()
        try:
            self._configuration = ConfigurationSnapshot.model_validate(configuration or {})
        except Exception:
            self._configuration = snapshot_from_settings(settings)

    async def invoke(
        self,
        prompt_name: str,
        output_model: type[TModel],
        *,
        payload: dict[str, Any],
    ) -> tuple[TModel, dict[str, Any]]:
        if not self._settings.llm_api_key:
            raise RuntimeError("format review model key is not configured")
        messages = self._prompts.render(
            prompt_name,
            "v1",
            payload=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )
        configuration = self._configuration.model.model_copy(
            update={"max_output_tokens": _OUTPUT_TOKEN_BUDGETS.get(prompt_name, 800)}
        )
        client = OpenAICompatibleClient(self._settings.llm_api_key)
        # Format findings are published only after schema and evidence gates.
        # Keep the provider call non-streaming; UI progress uses durable unit
        # events rather than unvalidated model tokens.
        result = await client.invoke_structured(messages, output_model, configuration)
        return result.output, asdict(result.metrics)
