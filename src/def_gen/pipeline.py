from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
import json


class Task(StrEnum):
    definer = "definer"
    feedback = "feedback"
    revisor = "revisor"
    judge = "judge"


@dataclass(slots=True)
class Step:
    name: str
    task: Task
    model: str
    prompt: str | None = None
    prompt_file: str | None = None
    schema: str | None = None
    schema_file: str | None = None
    post_field_renamer: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.prompt and self.prompt_file:
            raise ValueError("Only one of prompt or prompt_file should be provided.")
        if self.schema and self.schema_file:
            raise ValueError("Only one of schema or schema_file should be provided.")
        if not self.prompt and not self.prompt_file:
            raise ValueError("One of prompt or prompt_file should be provided.")
        
        self.prompt = self.prompt or Path(self.prompt_file).read_text(encoding="utf-8")
        self.schema = self.schema or json.loads(Path(self.schema_file).read_text(encoding="utf-8"))



def verify_steps(steps: list[Step]):
    # Names have to be unique
    names = [s.name for s in steps]
    if len(names) != len(set(names)):
        raise ValueError("Step names have to be unique!")
    
    if any(s.task not in Task for s in steps):
        raise ValueError(f"Step tasks must be one of {list(Task)}!")
    
    # First step has to be definer
    if steps[0].task != "definer":
        raise ValueError("First step has to be 'definer'!")
    
    for step_idx, step in enumerate(steps):
        # Feedback must be preceded by definer or revisor
        if step.task == "feedback":
            if steps[step_idx-1].task not in ["definer", "revisor", "feedback"]:
                raise ValueError("Feedback step must be preceded by definer, revisor, or feedback!")
        # Revisor must be preceded by feedback
        if step.task == "revisor":
            if steps[step_idx-1].task != "feedback":
                raise ValueError("Revisor step must be preceded by feedback!")
        # Judge must be preceded by definer or revisor
        if step.task == "judge":
            if steps[step_idx-1].task not in ["definer", "revisor"]:
                raise ValueError("Judge step must be preceded by definer or revisor!")