import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parents[2]


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
        raise ValueError(
            f"Failed to parse JSON: {json_exc.msg} at pos {json_exc.pos}; snippet={snippet!r}"
        ) from json_exc


class PathCompatibleEncoder(json.JSONEncoder):
    """
    JSON encoder that converts Path objects to strings.
    """

    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)
