from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import openai
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential


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
    except json.JSONDecodeError as e:
        snippet = text[:200].replace("\n", " ")
        raise ValueError(f"Failed to parse JSON: {e.msg} at pos {e.pos}; snippet={snippet!r}") from e


def llm_api_retry():
    return retry(
        retry=(
            retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
            | retry_if_exception_type(requests.exceptions.RequestException)
        ),
        wait=wait_random_exponential(multiplier=1, max=60),  # exponential backoff with jitter, up to 60s
        stop=stop_after_attempt(6),  # up to 6 attempts
        reraise=True,
    )
