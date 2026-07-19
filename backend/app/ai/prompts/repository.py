"""Versioned prompt loader with strict Jinja rendering."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..llm import ChatMessage


class PromptRepository:
    def __init__(self, root: Path | None = None) -> None:
        self._root = root or Path(__file__).resolve().parent
        self._environment = Environment(
            loader=FileSystemLoader(self._root),
            undefined=StrictUndefined,
            autoescape=False,
            keep_trailing_newline=True,
        )

    def render(self, name: str, version: str, **context: object) -> list[ChatMessage]:
        template = self._environment.get_template(f"{name}/{version}.jinja2")
        rendered = template.render(**context)
        delimiter = "\n---USER---\n"
        if delimiter not in rendered:
            raise ValueError(f"prompt {name}/{version} is missing the user delimiter")
        system, user = rendered.split(delimiter, maxsplit=1)
        return [
            ChatMessage(role="system", content=system.strip()),
            ChatMessage(role="user", content=user.strip()),
        ]

