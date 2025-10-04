import json
from typing import Any


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
        raise ValueError(f"Failed to parse JSON: {json_exc.msg} at pos {json_exc.pos}; snippet={snippet!r}") from json_exc
