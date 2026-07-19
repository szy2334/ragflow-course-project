from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator


SCHEMA_VERSION = "paper_chunk_v1"
PARSER_PIPELINE_VERSION = "user_paper_pipeline_v1"


_CONFIG_LOADED = False
_CONFIG_PATH: Path | None = None
_SENSITIVE_CONFIG_NAMES = frozenset(
    {
        "MINERU_TOKEN",
        "BAIDU_OCR_API_KEY",
        "BAIDU_OCR_SECRET_KEY",
        "RAGFLOW_API_KEY",
    }
)


class PipelineError(RuntimeError):
    pass


def load_user_paper_config(path: Path | str | None = None) -> Path:
    """Load local key=value settings without overriding real environment values.

    The default file is ``.env`` next to this module.  Set USER_PAPER_CONFIG
    to select another file.  The local file is deliberately Git-ignored.
    """
    global _CONFIG_LOADED, _CONFIG_PATH

    configured_path = path or os.environ.get("USER_PAPER_CONFIG")
    config_path = (
        Path(configured_path).expanduser()
        if configured_path
        else Path(__file__).resolve().with_name(".env")
    ).resolve()
    if _CONFIG_LOADED and _CONFIG_PATH == config_path:
        return config_path

    if config_path.exists():
        for line_number, raw_line in enumerate(
            config_path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            if "=" not in line:
                raise PipelineError(
                    f"Invalid configuration line at {config_path}:{line_number}"
                )
            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise PipelineError(
                    f"Invalid configuration key at {config_path}:{line_number}: {name!r}"
                )
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ.setdefault(name, value)

    _CONFIG_LOADED = True
    _CONFIG_PATH = config_path
    return config_path


def config_value(name: str, default: str | None = None) -> str | None:
    load_user_paper_config()
    return os.environ.get(name, default)


def config_path_value(name: str, default: str | None = None) -> Path | None:
    """Return a configured filesystem path relative to the configuration file.

    Command-line paths remain relative to the caller's working directory, while
    paths supplied by ``.env`` stay valid when a script is launched from a
    different directory.
    """
    value = config_value(name, default)
    if value is None or not value.strip():
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    config_path = load_user_paper_config()
    return (config_path.parent / path).resolve()


def config_bool(name: str, default: bool = False) -> bool:
    value = config_value(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise PipelineError(
        f"Configuration {name} must be true/false, 1/0, yes/no, or on/off"
    )


def safe_config_summary(names: Iterable[str] | None = None) -> dict[str, str]:
    """Return configuration presence only; secret values are never exposed."""
    load_user_paper_config()
    selected = names or sorted(_SENSITIVE_CONFIG_NAMES)
    return {
        name: "configured" if os.environ.get(name) else "missing"
        for name in selected
    }


def utc_now() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_uuid(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = value.replace("\u00ad", "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t\f\v]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def normalize_prose(text: str | None) -> str:
    value = normalize_text(text)
    # Join line-wrap hyphenation only when the next line starts with a lower-case letter.
    value = re.sub(r"(?<=[A-Za-z])[-‐]\n(?=[a-z])", "", value)
    value = re.sub(r"(?<![.!?:;\n])\n(?=[a-z])", " ", value)
    return value.strip()


def quality_flags(text: str, *, require_text: bool = False) -> list[str]:
    flags: list[str] = []
    if require_text and not text.strip():
        flags.append("empty_text")
    if "\ufffd" in text:
        flags.append("replacement_character")
    if re.search(r"(?:<sup>)?\?{2,}(?:</sup>)?", text):
        flags.append("unresolved_placeholder")
    if re.search(r"\b(?:[A-Za-z]\s){5,}[A-Za-z]\b", text):
        flags.append("possible_spaced_ocr_text")
    return flags


def estimate_tokens(text: str) -> int:
    ascii_words = len(re.findall(r"[A-Za-z0-9_]+", text))
    cjk_chars = len(re.findall(r"[\u3400-\u9fff]", text))
    symbols = len(re.findall(r"[^\w\s\u3400-\u9fff]", text))
    return max(1, ascii_words + (cjk_chars + 1) // 2 + symbols // 4)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PipelineError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise PipelineError(f"Expected JSON object at {path}:{line_number}")
            yield value


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
        Path(temp_name).replace(path)
    finally:
        temp_path = Path(temp_name)
        if temp_path.exists():
            temp_path.unlink()


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    rendered: list[str] = []
    count = 0
    for row in rows:
        rendered.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        count += 1
    atomic_write_text(path, "\n".join(rendered) + ("\n" if rendered else ""))
    return count


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise PipelineError(f"Required environment variable is not set: {name}")
    return value


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    return cleaned or "document"


def flatten_strings(value: Any) -> list[str]:
    output: list[str] = []
    if isinstance(value, str):
        if value.strip():
            output.append(value.strip())
    elif isinstance(value, dict):
        for child in value.values():
            output.extend(flatten_strings(child))
    elif isinstance(value, list):
        for child in value:
            output.extend(flatten_strings(child))
    return output


def recursive_values(value: Any, key: str) -> list[Any]:
    output: list[Any] = []
    if isinstance(value, dict):
        for current_key, child in value.items():
            if current_key == key:
                output.append(child)
            output.extend(recursive_values(child, key))
    elif isinstance(value, list):
        for child in value:
            output.extend(recursive_values(child, key))
    return output
