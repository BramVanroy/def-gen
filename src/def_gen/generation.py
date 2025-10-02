"""High-level generation utilities (plain, JSON mode, tool-calling).

This module provides a unified, typed API to perform text generation tasks
against an OpenAI-compatible chat-completions endpoint (OpenAI or vLLM).

Key features:
  - Plain text generation with system and user prompts
  - Strictly structured extraction using JSON mode
  - Strictly structured extraction using a required tool-call

All functions accept an :class:`def_gen.client.LLMClient` so callers can
configure the backend and model once.
"""

from __future__ import annotations

from typing import Any

from def_gen.client import LLMClient
from def_gen.utils import ChatMessage, make_messages, run_with_retry, safe_json_loads, function_tool_schema


def generate_text(
    client: LLMClient,
    *,
    messages: list[ChatMessage],
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    """Perform a plain chat completion and return assistant content.

    Args:
        client: Configured LLM client.
        messages: Chat messages including system and user roles.
        temperature: Sampling temperature.
        max_tokens: Token cap for the assistant reply.

    Returns:
        Assistant message content as a string.
    """

    def _call() -> str:
        resp = client.openai.chat.completions.create(
            model=client.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    return run_with_retry(_call)


def generate_json(
    client: LLMClient,
    *,
    messages: list[ChatMessage],
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> dict[str, Any]:
    """Generate a JSON object using the API's JSON mode and parse it.

    Args:
        client: Configured LLM client.
        messages: Chat messages culminating in a user request for JSON.
        temperature: Low temperature recommended for structure adherence.
        max_tokens: Token cap.

    Returns:
        Parsed JSON object.
    """

    def _call() -> dict[str, Any]:
        resp = client.openai.chat.completions.create(
            model=client.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content or "{}"
        return safe_json_loads(text)

    return run_with_retry(_call)


def generate_with_tool(
    client: LLMClient,
    *,
    messages: list[ChatMessage],
    tool_name: str,
    tool_description: str,
    tool_parameters_schema: dict[str, Any],
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> dict[str, Any]:
    """Generate structured output via a required tool-call and parse arguments.

    Args:
        client: Configured LLM client.
        messages: Conversation history instructing the model to call the tool.
        tool_name: Name of the function tool to expose.
        tool_description: Human-readable tool description.
        tool_parameters_schema: JSON schema for the function arguments.
        temperature: Typically 0.0 for deterministic structure.
        max_tokens: Token cap.

    Returns:
        Parsed tool call arguments as a dictionary.
    """

    tools = [
        function_tool_schema(
            name=tool_name,
            description=tool_description,
            parameters=tool_parameters_schema,
        )
    ]

    def _call() -> dict[str, Any]:
        resp = client.openai.chat.completions.create(
            model=client.model,
            messages=messages,
            tools=tools,
            tool_choice="required",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        call = resp.choices[0].message.tool_calls[0]
        return safe_json_loads(call.function.arguments)

    return run_with_retry(_call)


__all__ = [
    "generate_text",
    "generate_json",
    "generate_with_tool",
]
