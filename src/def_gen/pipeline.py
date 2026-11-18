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

    `definition_column` is required for judge steps (the column that is judged,
    exposed to the prompt as `te-beoordelen-definitie`) and rejected for other
    tasks. `order_rotation` is only allowed for ranker steps (see
    `PipelineStepConfig`). Judge and feedback steps without a `post_field_renamer` get
    `{field: f"{field}_{name}"}` for every property of their schema.
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
    last_of_task: bool = False
    post_field_renamer: dict[str, str] = field(default_factory=dict)
    client_init_kwargs: dict = field(default_factory=dict)
    client_run_kwargs: dict = field(default_factory=dict)
    definition_column: str | None = None
    order_rotation: int = 0

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

        if self.task == Task.judge and not self.definition_column:
            raise ValueError(
                f"Judge step {self.name!r} requires definition_column."
            )
        if self.task != Task.judge and self.definition_column:
            raise ValueError(
                "definition_column is only allowed for judge steps"
                f" (step {self.name!r})."
            )
        if self.task != Task.ranker and self.order_rotation:
            raise ValueError(
                "order_rotation is only allowed for ranker steps"
                f" (step {self.name!r})."
            )
        if (
            self.task in (Task.judge, Task.feedback)
            and not self.post_field_renamer
            and self.schema
        ):
            self.post_field_renamer = {
                f: f"{f}_{self.name}" for f in self.schema["properties"]
            }

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

        if not self.steps or self.steps[0].task != Task.definer:
            raise ValueError("Pipeline must start with a definer step!")
        if any(s.task == Task.definer for s in self.steps[1:]):
            raise ValueError("Pipeline can only contain one definer step!")

        saw_feedback = False
        for step in self.steps:
            if step.task == Task.feedback:
                saw_feedback = True
            elif step.task == Task.revisor and not saw_feedback:
                raise ValueError(
                    "Revisor step must be preceded by feedback somewhere!"
                )

        later_tasks: set[str] = set()
        for step in reversed(self.steps):
            step.last_of_task = step.task not in later_tasks
            later_tasks.add(step.task)

    def to_dict(self) -> dict:
        """Serialize all steps to a dict for JSON output."""
        return {"steps": [asdict(step) for step in self.steps]}
