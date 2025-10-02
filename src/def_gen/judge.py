"""LLM-as-a-judge utilities.

Provides functions to evaluate an output against a task using a rubric,
returning a structured verdict either via JSON mode or tool-calling.
"""

from __future__ import annotations

from typing import Any

from def_gen.client import LLMClient
from def_gen.utils import ChatMessage, make_messages
from def_gen.generation import generate_json, generate_with_tool


RUBRIC_SYSTEM = (
    "You are an impartial grader. Evaluate the OUTPUT against the TASK.\n"
    "Consider: correctness, completeness, clarity, and harmful content.\n"
    "Return JSON with: score (0-5 integer), label ('fail'|'poor'|'fair'|'good'|'excellent'), rationale (1-3 sentences).\n"
    "Scoring guidance:\n"
    "- 0-1: fail/poor, major errors or harmful content\n"
    "- 2: fair, noticeable issues\n"
    "- 3-4: good, minor issues\n"
    "- 5: excellent, correct and clear"
)


def judge_json(client: LLMClient, *, task: str, output: str) -> dict[str, Any]:
    """Judge using JSON mode.

    Args:
        client: Configured LLM client.
        task: Task description.
        output: Model-produced output to evaluate.

    Returns:
        A dict with keys ``score``, ``label``, and ``rationale``.
    """

    messages: list[ChatMessage] = make_messages(
        {"role": "system", "content": RUBRIC_SYSTEM},
        {"role": "user", "content": f"TASK:\n{task}\n\nOUTPUT:\n{output}\n\nProvide your evaluation."},
    )
    return generate_json(client, messages=messages, temperature=0.0, max_tokens=256)


def judge_tool(client: LLMClient, *, task: str, output: str) -> dict[str, Any]:
    """Judge using a required tool-call and parse arguments.

    Args:
        client: Configured LLM client.
        task: Task description.
        output: Model-produced output to evaluate.

    Returns:
        A dict with keys ``score``, ``label``, and ``rationale``.
    """

    messages: list[ChatMessage] = make_messages(
        {
            "role": "system",
            "content": "Evaluate against the rubric and call the tool with your structured verdict.",
        },
        {"role": "user", "content": f"TASK:\n{task}\n\nOUTPUT:\n{output}"},
    )

    schema = {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "minimum": 0, "maximum": 5},
            "label": {"type": "string", "enum": ["fail", "poor", "fair", "good", "excellent"]},
            "rationale": {"type": "string", "minLength": 1},
        },
        "required": ["score", "label", "rationale"],
        "additionalProperties": False,
    }

    return generate_with_tool(
        client,
        messages=messages,
        tool_name="submit_verdict",
        tool_description="Submit the numeric score, label, and rationale.",
        tool_parameters_schema=schema,
        temperature=0.0,
        max_tokens=256,
    )


__all__ = ["judge_json", "judge_tool"]
