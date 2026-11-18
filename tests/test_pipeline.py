from pathlib import Path

import pytest

from def_gen.pipeline import Pipeline, Step, Task


def _step(name: str, task: str, **kwargs) -> Step:
    if task == "judge":
        kwargs.setdefault("definition_column", "definitie-v1")
    return Step(
        name=name,
        task=task,
        model_name="model",
        provider="openai",
        full_prompt_template="prompt",
        **kwargs,
    )


def test_step_requires_exactly_one_prompt_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="One of prompt"):
        Step(name="a", task="definer", model_name="m", provider="openai")
    with pytest.raises(ValueError, match="Only one of prompt"):
        _step("a", "definer", file_directory=tmp_path)


def test_step_rejects_prefix_or_schema_with_file_directory(
    tmp_path: Path,
) -> None:
    kwargs = {
        "name": "a",
        "task": "definer",
        "model_name": "m",
        "provider": "openai",
        "file_directory": tmp_path,
    }
    with pytest.raises(ValueError, match="prefix"):
        Step(prompt_template_prefix="x", **kwargs)
    with pytest.raises(ValueError, match="schema"):
        Step(schema="{}", **kwargs)


def test_step_missing_directory_or_prompt_file(tmp_path: Path) -> None:
    kwargs = {
        "name": "a",
        "task": "definer",
        "model_name": "m",
        "provider": "openai",
    }
    with pytest.raises(ValueError, match="does not exist"):
        Step(file_directory=tmp_path / "missing", **kwargs)
    with pytest.raises(ValueError, match=r"specific\.md"):
        Step(file_directory=tmp_path, **kwargs)


def test_step_loads_prompt_prefix_and_schema_from_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / "shared.md").write_text("shared\n", encoding="utf-8")
    (tmp_path / "specific.md").write_text("specific", encoding="utf-8")
    (tmp_path / "schema.json").write_text(
        '{"type": "object"}', encoding="utf-8"
    )

    step = Step(
        name="a",
        task="definer",
        model_name="m",
        provider="openai",
        file_directory=str(tmp_path),
    )

    assert step.file_directory == tmp_path
    assert step.full_prompt_template == "shared\nspecific"
    assert step.schema == {"type": "object"}


def test_step_without_schema_or_shared_file(tmp_path: Path) -> None:
    (tmp_path / "specific.md").write_text("specific", encoding="utf-8")

    step = Step(
        name="a",
        task="definer",
        model_name="m",
        provider="openai",
        file_directory=tmp_path,
    )

    assert step.full_prompt_template == "specific"
    assert step.schema is None


def test_step_repr() -> None:
    step = _step("a", "definer", idx=2)

    assert repr(step) == (
        "Step(idx=2, name=a, task=definer, model_name=model)"
    )


def test_pipeline_assigns_idx_and_last_of_task() -> None:
    pipe = Pipeline(
        [
            _step("d", "definer"),
            _step("f1", "feedback"),
            _step("f2", "feedback"),
            _step("r", "revisor"),
        ]
    )

    assert [s.idx for s in pipe] == [0, 1, 2, 3]
    assert [s.last_of_task for s in pipe] == [True, False, True, True]
    assert len(pipe) == 4
    assert pipe[1].name == "f1"


def test_pipeline_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        Pipeline([_step("a", "definer"), _step("a", "feedback")])


def test_pipeline_rejects_unknown_task() -> None:
    with pytest.raises(ValueError, match="one of"):
        Pipeline([_step("a", "definer"), _step("b", "bogus")])


def test_pipeline_rejects_second_definer() -> None:
    with pytest.raises(ValueError, match="one definer"):
        Pipeline([_step("a", "definer"), _step("b", "definer")])


@pytest.mark.parametrize(
    "tasks",
    [
        [],
        ["feedback"],
        ["judge", "definer"],
        ["ranker", "definer", "judge"],
    ],
)
def test_pipeline_must_start_with_definer(tasks: list[str]) -> None:
    steps = [_step(f"s{i}", task) for i, task in enumerate(tasks)]

    with pytest.raises(ValueError, match="must start with a definer"):
        Pipeline(steps)


def test_pipeline_revisor_requires_feedback() -> None:
    with pytest.raises(ValueError, match="preceded by feedback"):
        Pipeline([_step("d", "definer"), _step("r", "revisor")])


def test_pipeline_marks_last_of_task_for_all_tasks() -> None:
    tasks = ["definer", "feedback", "feedback", "revisor", "judge", "ranker"]
    pipe = Pipeline([_step(f"s{i}", task) for i, task in enumerate(tasks)])

    assert [s.last_of_task for s in pipe] == [
        True,
        False,
        True,
        True,
        True,
        True,
    ]


def test_pipeline_contains_task_name_and_step() -> None:
    definer = _step("d", "definer")
    outsider = _step("x", "judge")
    pipe = Pipeline([definer, _step("j", "judge")])

    assert "definer" in pipe
    assert "judge" in pipe
    assert "ranker" not in pipe
    assert definer in pipe
    assert outsider not in pipe
    assert 3 not in pipe


def test_pipeline_to_dict_serializes_steps() -> None:
    pipe = Pipeline([_step("d", "definer"), _step("f", "feedback")])

    out = pipe.to_dict()

    assert [s["name"] for s in out["steps"]] == ["d", "f"]
    assert out["steps"][0]["task"] == Task.definer
    assert out["steps"][1]["idx"] == 1


def test_judge_step_requires_definition_column() -> None:
    with pytest.raises(ValueError, match="requires definition_column"):
        Step(
            name="j",
            task="judge",
            model_name="m",
            provider="openai",
            full_prompt_template="p",
        )


def test_definition_column_rejected_for_other_tasks() -> None:
    with pytest.raises(ValueError, match="only allowed for judge"):
        _step("r", "ranker", definition_column="definitie-v1")


@pytest.mark.parametrize("task", ["judge", "feedback"])
def test_empty_renamer_defaults_to_field_and_step_name(task: str) -> None:
    schema = {"properties": {"gebruik": {}, "betekenis": {}}}

    step = _step("s1", task, schema=schema)

    assert step.post_field_renamer == {
        "gebruik": "gebruik_s1",
        "betekenis": "betekenis_s1",
    }


def test_explicit_renamer_is_kept() -> None:
    step = _step(
        "s1",
        "feedback",
        schema={"properties": {"gebruik": {}}},
        post_field_renamer={"gebruik": "g"},
    )

    assert step.post_field_renamer == {"gebruik": "g"}
