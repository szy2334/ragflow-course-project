"""Shared mechanical lifecycle for structured model calls."""

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from .llm import StructuredLlm, StructuredModelResult
from .prompts import PromptRepository
from .schemas import ModelConfigSnapshot

TModel = TypeVar("TModel", bound=BaseModel)


class AgentRunner:
    """Render a versioned prompt and execute one controlled model call.

    The runner deliberately owns no business decisions.  Agents remain
    responsible for interpreting model output and building ``AgentResult``.
    """

    def __init__(self, llm: StructuredLlm, prompts: PromptRepository) -> None:
        self._llm = llm
        self._prompts = prompts

    async def run(
        self,
        *,
        prompt_name: str,
        prompt_version: str,
        output_model: type[TModel],
        model_config: ModelConfigSnapshot,
        context: dict[str, Any],
        on_content: Callable[[str], Awaitable[None]] | None = None,
    ) -> StructuredModelResult[TModel]:
        messages = self._prompts.render(prompt_name, prompt_version, **context)
        stream = getattr(self._llm, "invoke_structured_stream", None)
        if on_content is not None and callable(stream):
            return await stream(messages, output_model, model_config, on_content)
        return await self._llm.invoke_structured(messages, output_model, model_config)
