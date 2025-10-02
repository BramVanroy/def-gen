from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from def_gen.utils import ChatMessage, llm_api_retry, safe_json_loads


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

    def __init__(self, config: ModelConfig, name: str | None = None):
        self._config = config
        self._client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self.name = name or "undefined"

    def __str__(self) -> str:
        return f"LLMClient(name={self.name}, model={self.model}, base_url={self._config.base_url})"

    @property
    def client(self) -> OpenAI:
        """Return the underlying OpenAI client instance."""

        return self._client

    @property
    def model(self) -> str:
        """Default model identifier used for requests."""

        return self._config.model
    
    def is_openai(self) -> bool:
        """Heuristic to detect OpenAI's hosted API."""
        url = (self._config.base_url or "").lower()
        return "openai.com" in url

    @classmethod
    def from_env(cls) -> "LLMClient":
        """Create an ``LLMClient`` from environment variables.

        Returns:
            LLMClient: Configured client using environment defaults.
        """

        base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
        model = os.getenv("VLLMmodel", "meta-llama/Meta-Llama-3.1-8B-Instruct")
        return cls(ModelConfig(base_url=base_url, api_key=api_key, model=model))

    @llm_api_retry()
    def generate_text(
        self,
        *,
        messages: list[ChatMessage],
    ) -> str:
        """Perform a plain chat completion and return assistant content.

        Args:
            messages: Chat messages including system and user roles.

        Returns:
            Assistant message content as a string.
        """

        resp = self._client.chat.completions.create(            
            model=self.model,
            messages=messages,
        )
        return resp.choices[0].message.content or ""

    @llm_api_retry()
    def generate_json(
        self,
        json_schema: dict[str, Any],
        *,
        messages: list[ChatMessage],
    ) -> dict[str, Any]:
        """Generate a JSON object using the API's JSON mode and parse it.

        Args:
            json_schema: JSON schema defining the expected output structure.
            messages: Chat messages culminating in a user request for JSON.

        Returns:
            Parsed JSON object.
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_schema", "strict": True, "schema": json_schema},
        )
        
        text = resp.choices[0].message.content or "{}"
        return safe_json_loads(text)
