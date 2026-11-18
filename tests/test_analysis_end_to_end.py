import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import pytest
from datasets import Dataset, DatasetDict

from def_gen import definition_generation as dg
from def_gen.analysis.load import combine_runs, read_run, system_for_column
from def_gen.utils import ROOT


WRITERS = ["gemma4-26b-a4b", "qwen36-35b-a3b"]
RANKINGS = ["1 beter dan 2", "2 beter dan 1", "evenwaardig"]
NUM_ITEMS = 6
NUM_SCORED = 5
INVALID_ANW_ITEM = "pid-3"


def _load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        f"{name}_script", ROOT / "scripts" / f"{name}.py"
    )
    assert spec and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _number(*parts: object) -> int:
    text = "|".join(map(str, parts))
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


class _FakeAnnotator:
    """Fill the schema fields of a step with deterministic values per item.

    The prompt template is formatted with every row, so a template field
    that the dataset lacks fails here as it would with a real model.
    """

    prompts: ClassVar[dict[str, str]] = {}

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def annotate_dataset(self, **kwargs: Any) -> Dataset:
        ds = kwargs["dataset"]
        prefix = kwargs["task_prefix"]
        template = kwargs["prompt_template"]
        fields = list(kwargs["output_schema"]["properties"])

        def annotate(row: dict[str, Any]) -> dict[str, Any]:
            type(self).prompts[prefix] = template.format(**row)
            pid = row["betekenis_pid"]
            out: dict[str, Any] = {}
            if fields == ["definitie"]:
                out["definitie"] = f"{prefix} definitie van {row['lemmavorm']}"
            elif "ranking" in fields:
                out["argumentatie"] = "Omdat."
                out["ranking"] = RANKINGS[_number(prefix, pid) % 3]
            else:
                for field in fields:
                    out[field] = {
                        "analyse": "Goed.",
                        "score": 1 + _number(prefix, pid, field) % 5,
                    }
            invalid = (
                "judge-gemma4-on-anw" in prefix and pid == INVALID_ANW_ITEM
            )
            out[f"{prefix}valid_fields"] = not invalid
            out[f"{prefix}finish_reason"] = "length" if invalid else "stop"
            return out

        return ds.map(annotate)

    def destroy(self) -> None:
        return None


def _run_config(
    name: str, tmp_path: Path, source: Path, **overrides: Any
) -> None:
    data = json.loads(
        ROOT.joinpath("config", "study", f"{name}.json").read_text(
            encoding="utf-8"
        )
    )
    data |= {
        "root_output_dir": tmp_path / name,
        "from_hub_id": str(source),
        "num_proc": 1,
        **overrides,
    }
    dg.run_generation(dg.GenerateDefinitionsConfig.model_validate(data))


@pytest.fixture(scope="module")
def study_dirs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    tmp_path = tmp_path_factory.mktemp("study")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(ROOT)
        mp.setattr(dg, "Annotator", _FakeAnnotator)
        mp.setattr(
            dg,
            "_create_client_and_runtime_options",
            lambda step: (object(), object()),
        )
        items = Dataset.from_dict(
            {
                "betekenis_pid": [f"pid-{i}" for i in range(NUM_ITEMS)],
                "lemmavorm": [f"woord{i}" for i in range(NUM_ITEMS)],
                "voorbeelden": [
                    f"Een zin met woord{i}." for i in range(NUM_ITEMS)
                ],
                "originele-definitie": [
                    f"betekenis nummer {i}" for i in range(NUM_ITEMS)
                ],
            }
        )
        split = tmp_path / "anw-5ex"
        DatasetDict({"dev": items.select([0]), "eval": items}).save_to_disk(
            split
        )

        for writer in WRITERS:
            _run_config(f"writer-{writer}", tmp_path, split)
        _load_script("merge_writer_outputs").main(
            run_dirs=[str(tmp_path / f"writer-{w}") for w in WRITERS],
            output_dir=str(tmp_path / "head-to-head-input"),
        )
        merged = tmp_path / "head-to-head-input"
        _run_config("head-to-head", tmp_path, merged)
        _run_config("scores", tmp_path, merged, num_samples=NUM_SCORED)
    names = [*(f"writer-{w}" for w in WRITERS), "head-to-head", "scores"]
    return {name: tmp_path / name for name in names}


def test_tables_from_real_pipeline_runs(study_dirs: dict[str, Path]) -> None:
    data = combine_runs([read_run(d) for d in study_dirs.values()])

    scored = {f"pid-{i}" for i in range(NUM_SCORED)} - {INVALID_ANW_ITEM}
    assert set(data.scores["item"]) == scored
    assert set(data.pairs["item"]) == scored
    assert data.coverage["n_dropped"].max() == NUM_ITEMS - len(scored)
    assert set(data.scores["system"]) == {"anw", *WRITERS}
    assert set(data.scores["judge"]) == {
        "Qwen3.8-27B-FP8",
        "gemma-4-31B-it-qat-w4a16-ct",
        "gpt-oss-20b",
    }
    assert set(data.scores["criterion"]) == set(dg.JUDGE_FIELDS)
    assert data.scores["score"].between(1, 5).all()
    assert (
        data.scores.groupby(["system", "judge"]).size()
        == len(scored) * len(dg.JUDGE_FIELDS)
    ).all()
    assert data.scores.groupby(["system", "judge"]).ngroups == 9

    pair_groups = data.pairs.groupby(["system_a", "system_b"])["rotation"]
    assert {k: sorted(set(v)) for k, v in pair_groups} == {
        ("gemma4-26b-a4b", "anw"): [0, 1],
        ("qwen36-35b-a3b", "anw"): [0, 1],
        ("gemma4-26b-a4b", "qwen36-35b-a3b"): [0, 1],
    }
    assert (
        data.pairs.groupby(["system_a", "system_b"]).size()
        == len(scored) * 3 * 2
    ).all()
    assert set(data.definitions["system"]) == {"anw", *WRITERS}

    attrition = data.attrition.set_index(["run", "step"])
    invalid = attrition.loc[("scores", "judge-gemma4-on-anw")]
    assert (invalid["n_in"], invalid["n_invalid"]) == (NUM_SCORED, 1)


def test_scores_config_maps_columns_to_systems() -> None:
    config = json.loads(
        ROOT.joinpath("config", "study", "scores.json").read_text(
            encoding="utf-8"
        )
    )
    judge_steps = config["steps"]
    mapped = {
        (s["model_name"], system_for_column(s["definition_column"], None))
        for s in judge_steps
    }

    assert {system for _, system in mapped} == {"anw", *WRITERS}
    assert len({model for model, _ in mapped}) == 3
    assert len(mapped) == len(judge_steps) == 9
    assert len({s["name"] for s in judge_steps}) == 9
    renamed = [
        v for s in judge_steps for v in s["post_field_renamer"].values()
    ]
    assert len(renamed) == len(set(renamed))
    for step in judge_steps:
        system = system_for_column(step["definition_column"], None)
        assert step["name"].endswith(f"-on-{system}")
        assert all(
            v.startswith(f"{k}_{system}_judge_")
            for k, v in step["post_field_renamer"].items()
        )


def test_rotations_swap_the_shown_order(study_dirs: dict[str, Path]) -> None:
    pairs = read_run(study_dirs["head-to-head"]).pairs
    position = pairs.pivot_table(
        index=["judge", "item"], columns="rotation", values="position_a"
    )

    assert ((position[0] + position[1]) == 3).all()


def test_prompts_are_filled(study_dirs: dict[str, Path]) -> None:
    prompts = _FakeAnnotator.prompts

    assert any("woord" in p and "Definitie" in p for p in prompts.values())
    assert all("{" not in p.split("Woord:")[-1] for p in prompts.values())


def test_analyze_script_on_real_runs(
    study_dirs: dict[str, Path], tmp_path: Path
) -> None:
    out = tmp_path / "analysis"

    _load_script("analyze").main(
        run_dirs=[str(d) for d in study_dirs.values()],
        output_dir=str(out),
        num_bootstraps=20,
    )

    report = out.joinpath("report.md").read_text(encoding="utf-8")
    assert "## Bradley-Terry" in report
    assert "## Self-preference" in report
    bt = pd.read_csv(out / "bradley_terry.csv")
    assert set(bt["system"]) == {"anw", *WRITERS}
    assert bt["judge"].nunique() == 3
    families = pd.read_csv(out / "self_preference.csv")
    assert set(families["judge_family"]) == {"qwen", "gemma", "gpt-oss"}
    assert set(families["writer_family"]) == {"qwen", "gemma"}
