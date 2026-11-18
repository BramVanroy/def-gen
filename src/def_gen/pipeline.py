"""Pipeline steps and their order constraints."""

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from os import PathLike
from pathlib import Path
from typing import Literal


class Task(StrEnum):
    """Task types that a pipeline step can perform."""

    definer = "definer"
    feedback = "feedback"
    revisor = "revisor"
    judge = "judge"
    ranker = "ranker"


@dataclass(slots=True)
class Step:
    """One annotation step with its model, client settings and prompt.

    The prompt comes either from `full_prompt_template` or from a
    `file_directory` that holds `specific.md`, and optionally `shared.md`
    (prepended to the prompt) and `schema.json` (the output schema).
    """

    name: str
    task: Task
    model_name: str
    provider: Literal["openai", "anthropic", "vllm", "vllm-offline"]
    file_directory: str | PathLike | None = None
    full_prompt_template: str | None = None
    prompt_template_prefix: str = ""
    schema: str | None = None
    system_message: str | None = None
    idx: int = -1
    last_of_task: bool = None
    post_field_renamer: dict[str, str] = field(default_factory=dict)
    client_init_kwargs: dict = field(default_factory=dict)
    client_run_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.full_prompt_template and self.file_directory:
            raise ValueError(
                "Only one of prompt or prompt_file should be provided."
            )
        if self.prompt_template_prefix and self.file_directory:
            raise ValueError(
                "Only one of prefix or prompt_file should be provided."
            )
        if self.schema and self.file_directory:
            raise ValueError(
                "Only one of schema or schema_file should be provided."
            )
        if not self.full_prompt_template and not self.file_directory:
            raise ValueError(
                "One of prompt or prompt_file should be provided."
            )

        self.file_directory = (
            Path(self.file_directory) if self.file_directory else None
        )
        if self.file_directory:
            if not self.file_directory.exists():
                raise ValueError(
                    f"Provided file_directory {self.file_directory} does not exist!"
                )

            shared_file = self.file_directory / "shared.md"
            if shared_file.exists():
                self.prompt_template_prefix = shared_file.read_text(
                    encoding="utf-8"
                )

            prompt_file = self.file_directory / "specific.md"
            if not prompt_file.exists():
                raise ValueError(f"Prompt file {prompt_file} does not exist!")
            self.full_prompt_template = (
                self.prompt_template_prefix
                + prompt_file.read_text(encoding="utf-8")
            )

            schema_file = self.file_directory / "schema.json"
            if schema_file.exists():
                self.schema = json.loads(
                    Path(schema_file).read_text(encoding="utf-8")
                )

    def __repr__(self):
        return f"Step(idx={self.idx}, name={self.name}, task={self.task}, model_name={self.model_name})"


class Pipeline:
    """Ordered list of steps that is checked for a valid task order.

    Args:
        steps: The steps in execution order. Each step gets its index and
            whether it is the last step of its task.
    """

    def __init__(self, steps: list[Step]):
        self.steps = steps
        for idx, step in enumerate(self.steps):
            step.idx = idx
        self._verify_steps()

    def __iter__(self):
        return iter(self.steps)

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        return self.steps[idx]

    def __contains__(self, item: Step | str) -> bool:
        if isinstance(item, str):
            return any(step.task == item for step in self.steps)

        if isinstance(item, Step):
            return item in self.steps

        return False

    def _verify_steps(self):
        # Names have to be unique
        names = [s.name for s in self.steps]
        if len(names) != len(set(names)):
            raise ValueError("Step names have to be unique!")

        if any(s.task not in Task for s in self.steps):
            raise ValueError(f"Step tasks must be one of {list(Task)}!")

        saw_definer = False
        saw_feedback = False
        for step in self.steps:
            step_idx = step.idx
            if step.task == "definer":
                if saw_definer:
                    raise ValueError(
                        "Pipeline can only contain one definer step!"
                    )
                saw_definer = True
                if step_idx != 0:
                    raise ValueError("Definer step must be the first step!")
            elif step.task == "feedback":
                if not saw_definer:
                    raise ValueError(
                        "Feedback step must be preceded by definer somewhere!"
                    )
                saw_feedback = True
            elif step.task == "revisor":
                if not saw_feedback:
                    raise ValueError(
                        "Revisor step must be preceded by feedback somewhere!"
                    )
                if not saw_definer:
                    raise ValueError(
                        "Revisor step must be preceded by definer somewhere!"
                    )
            elif step.task == "judge" and not saw_definer:
                raise ValueError(
                    "Judge step must be preceded by definer somewhere!"
                )
            elif step.task == "ranker" and not saw_definer:
                raise ValueError(
                    "Ranker step must be preceded by definer somewhere!"
                )

        if not saw_definer:
            raise ValueError(
                "Pipeline must contain a definer step at the start!"
            )

        done_tasks = set(Task)
        for step in reversed(self.steps):
            if step.task in done_tasks:
                done_tasks.remove(step.task)
                step.last_of_task = True
            else:
                step.last_of_task = False
            if not done_tasks:
                break

    def to_dict(self) -> dict:
        """Serialize all steps to a dict for JSON output."""
        return {"steps": [asdict(step) for step in self.steps]}
