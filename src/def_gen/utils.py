from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(slots=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


def safe_json_loads(text: str) -> dict[str, Any]:
    """Parse JSON text, raising a ValueError on failure with context.

    Args:
        text: The JSON string to parse.

    Returns:
        Parsed JSON object.
    """

    try:
        return json.loads(text)
    except json.JSONDecodeError as json_exc:
        snippet = text[:200].replace("\n", " ")
        raise ValueError(f"Failed to parse JSON: {e.msg} at pos {e.pos}; snippet={snippet!r}") from json_exc
