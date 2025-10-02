"""Agentic iterative refinement utilities.

Implements a simple draft → critique → revise loop using a chat-completions
backend. The interface is small and typed to make it easy to integrate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from def_gen.client import LLMClient
from def_gen.utils import ChatMessage, make_messages
from def_gen.generation import generate_text


DEFAULT_SYSTEM = (
    "You are a writing assistant. Produce clear, factual, and concise content. "
    "When asked to revise, focus on correctness, clarity, and structure."
)

DEFAULT_CRITIC = (
    "You are a strict critic. Given a draft, list concrete issues and actionable fixes. "
    "Return a numbered list of improvements. If the draft is good enough, say 'NO MAJOR ISSUES'."
)

DEFAULT_REVISE = (
    "Revise the draft to address the listed issues. Keep the style neutral and concise. "
    "Do not add unrelated content."
)


class RefineResult(TypedDict):
    final: str
    critique: str
    iterations: int


@dataclass(slots=True)
class Refiner:
    """A small agent that drafts, critiques, and revises content.

    Attributes:
        client: LLM client wrapper with configured model.
        system_prompt: System prompt for drafting/revision steps.
        critic_prompt: System prompt for critique step.
        revise_instruction: Instruction merged with the draft and critique.
        temperature: Sampling temperature for all steps.
        max_tokens: Token cap for each step.
    """

    client: LLMClient
    system_prompt: str = DEFAULT_SYSTEM
    critic_prompt: str = DEFAULT_CRITIC
    revise_instruction: str = DEFAULT_REVISE
    temperature: float = 0.2
    max_tokens: int = 512

    def draft(self, topic: str, *, words: tuple[int, int] = (150, 200)) -> str:
        """Create an initial draft for a topic.

        Args:
            topic: The topic to write about.
            words: Min/max word suggestion for the draft.

        Returns:
            Draft text.
        """

        messages: list[ChatMessage] = make_messages(
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Write a concise explanation about: {topic} ({words[0]}-{words[1]} words).",
            },
        )
        return generate_text(self.client, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens)

    def critique(self, draft: str) -> str:
        """Critique a draft using the critic prompt."""

        messages: list[ChatMessage] = make_messages(
            {"role": "system", "content": self.critic_prompt},
            {"role": "user", "content": f"Draft:\n{draft}"},
        )
        return generate_text(self.client, messages=messages, temperature=0.0, max_tokens=self.max_tokens)

    def revise(self, draft: str, critique: str) -> str:
        """Revise a draft given a critique."""

        messages: list[ChatMessage] = make_messages(
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"{self.revise_instruction}\n\nDraft:\n{draft}\n\nIssues:\n{critique}",
            },
        )
        return generate_text(self.client, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens)

    def refine(self, topic: str, *, max_iters: int = 3) -> RefineResult:
        """Run draft → critique → revise for up to ``max_iters`` iterations.

        Stops early if the critique contains the sentinel 'NO MAJOR ISSUES'.
        """

        draft = self.draft(topic)
        last_critique = ""
        for i in range(max_iters):
            last_critique = self.critique(draft)
            if "NO MAJOR ISSUES" in last_critique.upper():
                return {"final": draft, "critique": last_critique, "iterations": i}
            draft = self.revise(draft, last_critique)
        return {"final": draft, "critique": last_critique, "iterations": max_iters}


__all__ = ["Refiner", "RefineResult"]
