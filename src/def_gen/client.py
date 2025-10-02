"""Client utilities for accessing OpenAI-compatible chat completions.

This module centralizes how we construct a client and pick a default model.
It supports both real OpenAI endpoints and vLLM servers that implement the
OpenAI-compatible API. Configuration is primarily read from environment
variables for convenience, but can also be provided directly.

Environment variables:
  - OPENAI_API_KEY: API key used by the OpenAI SDK (also for vLLM if needed)
  - VLLM_BASE_URL: Base URL of the vLLM/OpenAI-compatible server
                   (default: http://localhost:8000/v1)
  - VLLM_MODEL:    Model identifier to send in chat completion requests
                   (default: meta-llama/Meta-Llama-3.1-8B-Instruct)

Example:
  from def_gen.client import LLMClient

  client = LLMClient.from_env()
  resp = client.openai.chat.completions.create(
      model=client.model,
      messages=[{"role": "user", "content": "Hello"}],
  )
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass(slots=True)
class ModelConfig:
    """Model configuration for a chat-completions endpoint.

    Attributes:
        base_url: Base URL of the OpenAI-compatible server (OpenAI or vLLM).
        api_key: API key string (may be a dummy for local vLLM).
        model: The model identifier to use when creating completions.
    """

    base_url: str
    api_key: str
    model: str


class LLMClient:
    """Thin wrapper around the OpenAI client with model configuration.

    Provides a typed container that pairs the SDK client with a default model
    name so downstream code can stay DRY. The underlying client is accessible
    via the ``openai`` attribute.
    """

    def __init__(self, config: ModelConfig):
        self._config = config
        self._client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    @property
    def openai(self) -> OpenAI:
        """Return the underlying OpenAI client instance."""

        return self._client

    @property
    def model(self) -> str:
        """Default model identifier used for requests."""

        return self._config.model

    @classmethod
    def from_env(cls) -> "LLMClient":
        """Create an ``LLMClient`` from environment variables.

        Returns:
            LLMClient: Configured client using environment defaults.
        """

        base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
        model = os.getenv("VLLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
        return cls(ModelConfig(base_url=base_url, api_key=api_key, model=model))
