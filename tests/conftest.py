import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset

from def_gen.analysis.load import AnalysisData
from def_gen.definition_generation import JUDGE_FIELDS


def make_step(
    idx: int, name: str, task: str, model_name: str, **kwargs: Any
) -> dict[str, Any]:
    step = {
        "idx": idx,
        "name": name,
        "task": task,
        "model_name": model_name,
        "post_field_renamer": {},
        "definition_column": None,
        "order_rotation": 0,
    }
    if task == "judge":
        step["post_field_renamer"] = {f: f"{f}_{name}" for f in JUDGE_FIELDS}
    return {**step, **kwargs}


def write_run(
    run_dir: Path,
    steps: list[dict[str, Any]],
    columns: dict[str, list],
    attrition: dict[str, dict] | None = None,
) -> Path:
    """Write a run directory in the layout of `run_generation`."""
    run_dir.mkdir(parents=True)
    run_dir.joinpath("pipeline.json").write_text(
        json.dumps({"steps": steps}), encoding="utf-8"
    )
    Dataset.from_dict(columns).save_to_disk(run_dir / "full_process_done")
    for step in steps:
        if attrition and step["name"] in attrition:
            step_dir = (
                run_dir / f"{step['idx'] + 1}-{step['task']}-{step['name']}"
            )
            step_dir.mkdir()
            step_dir.joinpath("attrition.json").write_text(
                json.dumps(attrition[step["name"]]), encoding="utf-8"
            )
    return run_dir


def simulate_study(
    num_items: int = 60,
    seed: int = 0,
    writers: dict[str, float] | None = None,
    position_bias: float = 0.0,
) -> AnalysisData:
    """Simulate scores and two-way rankings of writers against ANW.

    Each writer has a quality offset relative to ANW (0 = equal). Three
    judges score every system on all criteria (1-5) and rank each writer
    against ANW in both orders. `position_bias` raises the log-odds that the
    definition at position 1 wins.
    """
    rng = np.random.default_rng(seed)
    writers = writers or {"gemma-3-27b": 0.8, "Qwen3.6-27B": -0.4}
    judges = ["Qwen3.6-27B", "gemma-3-27b", "gpt-oss-120b"]
    quality = {"anw": rng.normal(0, 1, num_items)}
    for writer, offset in writers.items():
        quality[writer] = quality["anw"] + offset + rng.normal(0, 1, num_items)

    score_rows, pair_rows, def_rows = [], [], []
    for system, q in quality.items():
        for item in range(num_items):
            words = max(3, int(8 + 2 * q[item] + rng.normal(0, 2)))
            def_rows.append(
                {
                    "item": item,
                    "system": system,
                    "definition": " ".join(["woord"] * words),
                }
            )
        for judge in judges:
            for criterion in JUDGE_FIELDS:
                noise = rng.normal(0, 0.8, num_items)
                scores = np.clip(np.round(3.5 + q + noise), 1, 5)
                score_rows += [
                    {
                        "run": "sim",
                        "item": item,
                        "system": system,
                        "judge": judge,
                        "criterion": criterion,
                        "score": int(s),
                    }
                    for item, s in enumerate(scores)
                ]
    for writer in writers:
        for judge in judges:
            for rotation in (0, 1):
                for item in range(num_items):
                    position_a = 1 + (item + rotation) % 2
                    diff = quality[writer][item] - quality["anw"][item]
                    diff += position_bias * (1 if position_a == 1 else -1)
                    u = rng.random()
                    p_a = 1 / (1 + np.exp(-2 * diff))
                    if abs(diff) < 0.15 and u < 0.5:
                        outcome = "tie"
                    else:
                        outcome = "a" if rng.random() < p_a else "b"
                    winner = {
                        "a": position_a,
                        "b": 3 - position_a,
                        "tie": None,
                    }[outcome]
                    pair_rows.append(
                        {
                            "run": "sim",
                            "item": item,
                            "judge": judge,
                            "rotation": rotation,
                            "system_a": writer,
                            "system_b": "anw",
                            "position_a": position_a,
                            "outcome": outcome,
                            "winner_position": winner,
                        }
                    )
    return AnalysisData(
        scores=pd.DataFrame(score_rows),
        pairs=pd.DataFrame(pair_rows),
        definitions=pd.DataFrame(def_rows),
        attrition=pd.DataFrame(),
    )


@pytest.fixture(scope="session")
def study() -> AnalysisData:
    return simulate_study()
