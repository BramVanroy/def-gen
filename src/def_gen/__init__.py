"""def_gen: LLM generation and evaluation helpers.

This package provides a small, typed, and DRY API for:
	- Plain text and structured generation (JSON mode, tool-calling)
	- LLM-as-a-judge verdicts (JSON or tool)
	- An iterative agent that drafts, critiques, and revises content

The library targets OpenAI-compatible chat-completions endpoints, including
OpenAI itself and local vLLM servers.
"""

from .client import LLMClient, ModelConfig, get_client, get_model
from .generation import generate_json, generate_text, generate_with_tool
from .judge import judge_json, judge_tool
from .agent import Refiner, RefineResult

__all__ = [
		# Client
		"LLMClient",
		"ModelConfig",
		"get_client",
		"get_model",
		# Generation
		"generate_text",
		"generate_json",
		"generate_with_tool",
		# Judge
		"judge_json",
		"judge_tool",
		# Agent
		"Refiner",
		"RefineResult",
]
