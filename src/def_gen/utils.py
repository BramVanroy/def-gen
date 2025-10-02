"""Shared utilities for the def_gen package.

This module contains small helpers to keep other modules DRY:
  - Typed aliases for Chat messages
  - Safe JSON parsing
  - Tool schema builders for OpenAI tool-calling
  - Retry decorator configured for chat completions

All functions follow PEP8 and include type hints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Literal, Optional, TypedDict

from tenacity import Retrying, stop_after_attempt, wait_exponential


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


def make_messages(*msgs: ChatMessage) -> list[ChatMessage]:
    """Create a list of chat messages with validation.

    Args:
        *msgs: Chat messages in order.

    Returns:
        A list of messages suitable for OpenAI chat.completions.
    """

    return list(msgs)


def safe_json_loads(text: str) -> dict[str, Any]:
    """Parse JSON text, raising a ValueError on failure with context.

    Args:
        text: The JSON string to parse.

    Returns:
        Parsed JSON object.
    """

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        snippet = text[:200].replace("\n", " ")
        raise ValueError(f"Failed to parse JSON: {e.msg} at pos {e.pos}; snippet={snippet!r}") from e


def function_tool_schema(
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Build a single function tool schema for OpenAI tool-calling.

    Args:
        name: Function/tool name.
        description: What the tool does.
        parameters: JSON schema for the tool arguments.

    Returns:
        A ``tools`` entry suitable for ``tools=[...]``.
    """

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def default_retryer(max_attempts: int = 3) -> Retrying:
    """Return a Tenacity Retrying instance for transient chat failures.

    Args:
        max_attempts: Max retry attempts.

    Returns:
        A configured ``Retrying`` object with exponential backoff.
    """

    return Retrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )


def run_with_retry(callable_fn: Callable[[], Any], *, max_attempts: int = 3) -> Any:
    """Execute a callable with retry on exceptions.

    Args:
        callable_fn: Zero-arg callable to execute.
        max_attempts: Max retry attempts.

    Returns:
        The callable's return value.
    """

    for attempt in default_retryer(max_attempts=max_attempts):
        with attempt:
            return callable_fn()
