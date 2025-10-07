import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class Task(StrEnum):
    definer = "definer"
    feedback = "feedback"
    revisor = "revisor"
    judge = "judge"


@dataclass(slots=True)
class Step:
    name: str
    task: Task
    model_name: str
    prompt: str | None = None
    prompt_file: str | None = None
    schema: str | None = None
    schema_file: str | None = None
    idx: int = -1
    last_of_task: bool = None
    post_field_renamer: dict[str, str] = field(default_factory=dict)
    anno_init_kwargs: dict = field(default_factory=dict)
    anno_run_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.prompt and self.prompt_file:
            raise ValueError("Only one of prompt or prompt_file should be provided.")
        if self.schema and self.schema_file:
            raise ValueError("Only one of schema or schema_file should be provided.")
        if not self.prompt and not self.prompt_file:
            raise ValueError("One of prompt or prompt_file should be provided.")

        self.prompt = self.prompt or Path(self.prompt_file).read_text(encoding="utf-8")
        self.schema = self.schema or json.loads(Path(self.schema_file).read_text(encoding="utf-8"))


class Pipeline:
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

    def _verify_steps(self):
        # Names have to be unique
        names = [s.name for s in self.steps]
        if len(names) != len(set(names)):
            raise ValueError("Step names have to be unique!")

        if any(s.task not in Task for s in self.steps):
            raise ValueError(f"Step tasks must be one of {list(Task)}!")

        # First step has to be definer
        if self.steps[0].task != "definer":
            raise ValueError("First step has to be 'definer'!")

        for step in self.steps:
            step_idx = step.idx
            # Feedback must be preceded by definer or revisor
            if step.task == "feedback":
                if self.steps[step_idx - 1].task not in ["definer", "revisor", "feedback"]:
                    raise ValueError("Feedback step must be preceded by definer, revisor, or feedback!")
            # Revisor must be preceded by feedback
            if step.task == "revisor":
                if self.steps[step_idx - 1].task != "feedback":
                    raise ValueError("Revisor step must be preceded by feedback!")
            # Judge must be preceded by definer or revisor
            if step.task == "judge":
                if self.steps[step_idx - 1].task not in ["definer", "revisor", "judge"]:
                    raise ValueError("Judge step must be preceded by another judge, definer or revisor!")

        # Cannot have more than one definer
        if sum(1 for s in self.steps if s.task == "definer") > 1:
            raise ValueError("Pipeline can only contain one definer step!")

        done_tasks = set(Task)
        for step in reversed(self.steps):
            if step.task in done_tasks:
                done_tasks.remove(step.task)
                step.last_of_task = True
            else:
                step.last_of_task = False
            if not done_tasks:
                break
