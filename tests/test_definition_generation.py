import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import llm_annotator
import numpy as np
import pytest
from datasets import Dataset
from pydantic import ValidationError

from def_gen import definition_generation as dg


def _make_step(**kwargs):
    defaults = {
        "idx": 0,
        "name": "step-a",
        "task": "definer",
        "model_name": "RedHatAI/gemma-3-27b-it-FP8-dynamic",
        "provider": None,
        "post_field_renamer": {"definitie": "definitie-v1"},
        "client_init_kwargs": {},
        "client_run_kwargs": {},
        "full_prompt_template": "prompt",
        "prompt_template_prefix": "",
        "schema": None,
        "system_message": None,
        "last_of_task": True,
        "order_rotation": 0,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_config_coerces_path_and_defaults() -> None:
    config = dg.GenerateDefinitionsConfig(root_output_dir="outputs/custom")

    assert config.root_output_dir == Path("outputs/custom")
    assert config.batch_size == 256


def test_config_rejects_removed_or_unknown_values() -> None:
    with pytest.raises(ValidationError):
        dg.GenerateDefinitionsConfig(gpu_memory_utilization=1.5)
    with pytest.raises(ValidationError):
        dg.GenerateDefinitionsConfig(workflow="no-revision")


def test_from_sources_cli_overrides_config_file() -> None:
    config = dg.GenerateDefinitionsConfig.from_sources(
        config_data={"num_samples": 10, "batch_size": 4},
        overrides={"num_samples": 2},
    )

    assert config.num_samples == 2
    assert config.batch_size == 4


def test_pipeline_step_config_requires_prompt_source() -> None:
    with pytest.raises(ValidationError):
        dg.PipelineStepConfig(
            name="step-a",
            task="definer",
            model_name="fake",
        )


def test_pipeline_step_config_client_kwargs_accepted() -> None:
    step = dg.PipelineStepConfig.model_validate(
        {
            "name": "custom-definer",
            "task": "definer",
            "model_name": "gpt-4o-mini",
            "full_prompt_template": "Prompt",
            "provider": "openai",
            "client_init_kwargs": {"api_key": "x"},
            "client_run_kwargs": {"max_completion_tokens": 12},
            "schema": {"type": "object"},
        }
    )

    out = step.to_step()

    assert step.output_schema == {"type": "object"}
    assert out.provider == "openai"
    assert out.client_init_kwargs["api_key"] == "x"
    assert out.client_run_kwargs["max_completion_tokens"] == 12


def test_set_seed_seeds_random_and_numpy() -> None:
    dg.set_seed(123)
    first = (random.random(), np.random.rand())
    dg.set_seed(123)

    assert (random.random(), np.random.rand()) == first


@pytest.mark.parametrize("cuda_available", [True, False])
def test_set_seed_seeds_torch_when_installed(
    monkeypatch: pytest.MonkeyPatch, cuda_available: bool
) -> None:
    calls: list[tuple[str, int]] = []
    fake_torch = SimpleNamespace(
        manual_seed=lambda seed: calls.append(("cpu", seed)),
        cuda=SimpleNamespace(
            is_available=lambda: cuda_available,
            manual_seed_all=lambda seed: calls.append(("cuda", seed)),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    dg.set_seed(7)

    expected = [("cpu", 7), ("cuda", 7)] if cuda_available else [("cpu", 7)]
    assert calls == expected


def test_set_seed_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)

    dg.set_seed(7)


@pytest.mark.parametrize(
    ("provider", "client_cls", "options_cls", "run_kwargs"),
    [
        (
            "openai",
            "OpenAIClient",
            "OpenAIRuntimeOptions",
            {"max_completion_tokens": 333},
        ),
        (
            "anthropic",
            "ClaudeClient",
            "ClaudeRuntimeOptions",
            {"max_completion_tokens": 333},
        ),
        (
            "vllm",
            "VLLMOnlineClient",
            "VLLMOnlineRuntimeOptions",
            {"max_completion_tokens": 333},
        ),
        (
            "vllm-offline",
            "VLLMOfflineClient",
            "VLLMOfflineRuntimeOptions",
            {"max_completion_tokens": 333},
        ),
    ],
)
def test_create_client_and_runtime_options_per_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    client_cls: str,
    options_cls: str,
    run_kwargs: dict,
) -> None:
    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(llm_annotator, client_cls, FakeClient)
    monkeypatch.setattr(llm_annotator, options_cls, FakeOptions)
    step = _make_step(
        model_name="some-model",
        provider=provider,
        client_init_kwargs={"api_key": "x"},
        client_run_kwargs=run_kwargs,
    )

    client, options = dg._create_client_and_runtime_options(step)

    assert isinstance(client, FakeClient)
    assert client.kwargs == {"model": "some-model", "api_key": "x"}
    assert isinstance(options, FakeOptions)
    assert options.kwargs == run_kwargs


def test_create_client_rejects_unsupported_provider() -> None:
    step = _make_step(provider=None)

    with pytest.raises(ValueError, match="provider"):
        dg._create_client_and_runtime_options(step)


def test_concat_feedback_skips_repeated_characters() -> None:
    sample = {
        "revisie-nodig_a": True,
        "gebruik_a": "goede feedback",
        "structuur_a": "aaaaa",
        "lemmavorm": "woord",
    }
    feedback_cols = {
        "feedback-a": ["revisie-nodig_a", "gebruik_a", "structuur_a"]
    }

    out = dg.concat_feedback(sample, feedback_cols)

    assert "goede feedback" in out["feedback"]
    assert "aaaaa" not in out["feedback"]


def test_prepare_examples_formats_context() -> None:
    out = dg.prepare_examples(
        [
            "Dit is een test met Woord in het midden en extra tekst.",
            "Nog een voorbeeldzin met woord erin.",
        ],
        word="woord",
        num_examples=1,
        context_words=2,
    )

    assert out["voorbeelden"].startswith("- ")
    assert "woord" in out["voorbeelden"].lower()


def test_shuffle_definitions_is_deterministic() -> None:
    sample = {
        "definitie-v1": "v1",
        "definitie-v2": "v2",
        "originele-definitie": "orig",
        "id": 1,
    }

    first = dg.shuffle_definitions(sample)
    second = dg.shuffle_definitions(sample)

    assert first == second


def test_check_rank_validity() -> None:
    sample_ok = {"c1": "Definitie 1", "c2": "Definitie 2", "c3": "Definitie 3"}
    sample_bad = {
        "c1": "Definitie 1",
        "c2": "Definitie 1",
        "c3": "Definitie 3",
    }

    assert dg.check_rank_validity(sample_ok, ["c1", "c2", "c3"])
    assert not dg.check_rank_validity(sample_bad, ["c1", "c2", "c3"])

    with pytest.raises(ValueError):
        dg.check_rank_validity(sample_ok, ["c1", "c2"])


def test_build_pipeline_uses_custom_steps() -> None:
    config = dg.GenerateDefinitionsConfig(
        steps=[
            dg.PipelineStepConfig(
                name="custom-definer",
                task="definer",
                model_name="gpt-4o-mini",
                file_directory=Path("instructions/definer"),
                provider="openai",
                client_init_kwargs={"api_key": "token"},
                client_run_kwargs={"max_completion_tokens": 42},
                post_field_renamer={"definitie": "definitie-v1"},
            )
        ]
    )

    pipe = dg.build_pipeline(config)

    assert [step.name for step in pipe] == ["custom-definer"]
    assert pipe[0].provider == "openai"
    assert pipe[0].client_run_kwargs["max_completion_tokens"] == 42


class _RunDataset:
    pushed = None

    def __init__(self, rows):
        self.rows = rows

    @property
    def column_names(self):
        return list(self.rows[0].keys()) if self.rows else []

    def load_from_disk(self, path):
        rows_file = Path(path) / "rows.json"
        if rows_file.exists():
            return _RunDataset(json.loads(rows_file.read_text("utf-8")))
        return self

    def filter(self, fn, input_columns=None, num_proc=None):
        key = input_columns[0]
        filtered = [row for row in self.rows if fn(row.get(key))]
        return _RunDataset(filtered)

    def map(self, fn, num_proc=None, fn_kwargs=None, remove_columns=None):
        fn_kwargs = fn_kwargs or {}
        new_rows = []
        for row in self.rows:
            payload = dict(row)
            payload.update(fn(payload, **fn_kwargs))
            if remove_columns:
                for col in remove_columns:
                    payload.pop(col, None)
            new_rows.append(payload)
        return _RunDataset(new_rows)

    def rename_columns(self, mapping):
        rows = []
        for row in self.rows:
            payload = dict(row)
            for old, new in mapping.items():
                payload[new] = payload.pop(old)
            rows.append(payload)
        return _RunDataset(rows)

    def add_column(self, col, values):
        rows = []
        for row, value in zip(self.rows, values, strict=True):
            payload = dict(row)
            payload[col] = value
            rows.append(payload)
        return _RunDataset(rows)

    @property
    def features(self):
        return self.column_names

    def select_columns(self, columns):
        return _RunDataset([{c: row[c] for c in columns} for row in self.rows])

    def cast(self, features):
        return self

    def sort(self, column):
        return _RunDataset(sorted(self.rows, key=lambda row: row[column]))

    def remove_columns(self, columns):
        return _RunDataset(
            [
                {k: v for k, v in row.items() if k not in columns}
                for row in self.rows
            ]
        )

    def to_list(self):
        return [dict(row) for row in self.rows]

    def save_to_disk(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        path.joinpath(".saved").write_text("1", encoding="utf-8")
        path.joinpath("rows.json").write_text(
            json.dumps(self.rows), encoding="utf-8"
        )

    def push_to_hub(self, hub_id, private=True):
        _RunDataset.pushed = (hub_id, private)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.rows[key]
        return [row[key] for row in self.rows]

    def __len__(self):
        return len(self.rows)


class _FakePipeline:
    def __init__(self, steps):
        self.steps = steps

    def __iter__(self):
        return iter(self.steps)

    def __contains__(self, item):
        if isinstance(item, str):
            return any(step.task == item for step in self.steps)
        return item in self.steps

    def to_dict(self):
        return {
            "steps": [
                {"name": step.name, "task": step.task} for step in self.steps
            ]
        }


class _FakeAnnotator:
    init_calls: ClassVar[list] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeAnnotator.init_calls.append(kwargs)

    def annotate_dataset(self, **kwargs):
        ds = kwargs["dataset"]
        task_prefix = kwargs["task_prefix"]
        rows = []
        for row in ds.rows:
            payload = dict(row)
            payload[f"{task_prefix}valid_fields"] = True
            payload[f"{task_prefix}finish_reason"] = "stop"
            rows.append(payload)
        return _RunDataset(rows)

    def destroy(self):
        return None


def test_run_generation_smoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _FakeAnnotator.init_calls = []
    base_ds = _RunDataset(
        [{"definitie": "Een voorbeeld definitie", "feedback": ""}]
    )

    fake_step = _make_step(name="definer-fake")

    monkeypatch.setattr(dg, "set_seed", lambda seed: None)
    monkeypatch.setattr(dg, "init_data", lambda **kwargs: base_ds)
    monkeypatch.setattr(
        dg, "build_pipeline", lambda config: _FakePipeline([fake_step])
    )
    monkeypatch.setattr(dg, "Annotator", _FakeAnnotator)
    monkeypatch.setattr(
        dg,
        "_create_client_and_runtime_options",
        lambda step: (object(), object()),
    )

    config = dg.GenerateDefinitionsConfig(
        root_output_dir=tmp_path,
        hub_id=None,
        batch_size=8,
    )
    dg.run_generation(config)

    assert (tmp_path / "pipeline.json").exists()
    assert (tmp_path / "1-definer-definer-fake" / "done" / ".saved").exists()
    assert (tmp_path / "full_process_done" / ".saved").exists()
    assert _FakeAnnotator.init_calls[0]["batch_size"] == 8


def test_run_generation_pushes_formatted_hub_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _RunDataset.pushed = None
    base_ds = _RunDataset([{"definitie": "x", "feedback": ""}])
    fake_step = _make_step(name="definer-fake")

    monkeypatch.setattr(dg, "set_seed", lambda seed: None)
    monkeypatch.setattr(dg, "init_data", lambda **kwargs: base_ds)
    monkeypatch.setattr(
        dg, "build_pipeline", lambda config: _FakePipeline([fake_step])
    )
    monkeypatch.setattr(dg, "Annotator", _FakeAnnotator)
    monkeypatch.setattr(
        dg,
        "_create_client_and_runtime_options",
        lambda step: (object(), object()),
    )

    config = dg.GenerateDefinitionsConfig(
        root_output_dir=tmp_path,
        hub_id="owner/ds-{num_samples}",
    )
    dg.run_generation(config)

    assert _RunDataset.pushed == ("owner/ds-1", True)


def test_concat_feedback_numbers_reviewers_and_skips_no_revision() -> None:
    sample = {
        "revisie-nodig_a": False,
        "gebruik_a": "niet nodig",
        "revisie-nodig_b": True,
        "gebruik_b": "te vaag",
        "structuur_b": None,
        "lemmavorm": "woord",
    }
    feedback_cols = {
        "fb-a": ["revisie-nodig_a", "gebruik_a"],
        "fb-b": ["revisie-nodig_b", "gebruik_b", "structuur_b"],
    }

    out = dg.concat_feedback(sample, feedback_cols)

    assert out == {"feedback": "*Feedback van reviewer 2*\n- gebruik: te vaag"}


def test_concat_feedback_is_empty_when_no_reviewer_needs_revision() -> None:
    sample = {"revisie-nodig_a": False, "lemmavorm": "woord"}

    out = dg.concat_feedback(sample, {"fb-a": ["revisie-nodig_a"]})

    assert out == {"feedback": ""}


def test_prepare_examples_truncates_context_with_markers() -> None:
    out = dg.prepare_examples(
        ["een twee drie woord vijf zes zeven"],
        word="woord",
        num_examples=3,
        context_words=1,
    )

    assert out == {"voorbeelden": "- [...] drie woord vijf [...]"}


def test_prepare_examples_limits_number_of_examples() -> None:
    out = dg.prepare_examples(
        ["woord a", "woord b", "woord c"], word="woord", num_examples=2
    )

    assert len(out["voorbeelden"].splitlines()) == 2


def test_shuffle_definitions_two_candidates() -> None:
    sample = {"definitie-v1": "v1", "originele-definitie": "orig"}

    out = dg.shuffle_definitions(sample)

    assert sorted(out["shuffle-order"]) == [
        "definitie-v1",
        "originele-definitie",
    ]
    assert "random-definitie-3" not in out
    mapping = {"definitie-v1": "v1", "originele-definitie": "orig"}
    assert out["random-definitie-1"] == mapping[out["shuffle-order"][0]]
    assert out["random-definitie-2"] == mapping[out["shuffle-order"][1]]


def test_shuffle_definitions_three_candidates() -> None:
    sample = {
        "definitie-v1": "v1",
        "definitie-v2": "v2",
        "originele-definitie": "orig",
    }
    mapping = {
        "definitie-v1": "v1",
        "definitie-v2": "v2",
        "originele-definitie": "orig",
    }

    out = dg.shuffle_definitions(sample)

    assert sorted(out["shuffle-order"]) == sorted(mapping)
    for i, name in enumerate(out["shuffle-order"], 1):
        assert out[f"random-definitie-{i}"] == mapping[name]


def test_check_rank_validity_ignores_missing_values() -> None:
    sample = {"c1": "Definitie 1", "c2": None, "c3": "Definitie 3"}
    unknown = {"c1": "Definitie 1", "c2": "Definitie 9", "c3": "Definitie 3"}

    assert not dg.check_rank_validity(sample, ["c1", "c2", "c3"])
    assert not dg.check_rank_validity(unknown, ["c1", "c2", "c3"])


def _anw_row(lemma: str, **overrides) -> dict:
    row = {
        "lemmavorm": lemma,
        "woordsoort": "substantief",
        "definitie": "een lange definitie met genoeg woorden erin",
        "voorbeelden": [f"een {lemma} staat hier", f"daar ligt de {lemma}"],
        "artikel_meerdere_betekenissen": False,
        "mini_definitie": "mini",
        "artikel_id": 1,
        "betekenis_id": 1,
        "artikel_laatst_bewerkt": 2020,
    }
    row.update(overrides)
    return row


def test_init_data_filters_and_prepares_rows(tmp_path: Path) -> None:
    rows = [
        _anw_row("boom"),
        _anw_row("huis"),
        _anw_row("lopen", woordsoort="werkwoord"),
        _anw_row("kort", definitie="te kort"),
        _anw_row("Utrecht"),
        _anw_row("poly", artikel_meerdere_betekenissen=True),
        _anw_row("ook", definitie="een lange definitie ook: nog iets erbij"),
        _anw_row("weinig", voorbeelden=["alleen een weinig", "geen match"]),
    ]
    path = tmp_path / "data.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    ds = dg.init_data(
        num_samples=None, min_num_examples=2, num_proc=None, data_path=path
    )

    assert sorted(ds["lemmavorm"]) == ["boom", "huis"]
    assert "originele-definitie" in ds.column_names
    assert "definitie" not in ds.column_names
    assert "woordsoort" not in ds.column_names
    assert all(isinstance(v, str) for v in ds["artikel_laatst_bewerkt"])
    for lemma, voorbeelden in zip(
        ds["lemmavorm"], ds["voorbeelden"], strict=True
    ):
        assert len(voorbeelden.splitlines()) == 2
        assert lemma in voorbeelden


def test_init_data_limits_num_samples(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    rows = [_anw_row(w) for w in ["boom", "huis", "tafel"]]
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    ds = dg.init_data(num_samples=2, min_num_examples=2, data_path=path)

    assert len(ds) == 2


class _ScriptedAnnotator(_FakeAnnotator):
    """Annotator that adds fixed columns and marks rows invalid by index."""

    new_columns: ClassVar[dict] = {}
    invalid_rows: ClassVar[set[int]] = set()
    num_calls = 0

    def annotate_dataset(self, **kwargs):
        type(self).num_calls += 1
        task_prefix = kwargs["task_prefix"]
        rows = []
        for i, row in enumerate(kwargs["dataset"].rows):
            payload = dict(row)
            payload.update(type(self).new_columns)
            valid = i not in type(self).invalid_rows
            payload[f"{task_prefix}valid_fields"] = valid
            payload[f"{task_prefix}finish_reason"] = (
                "stop" if valid else "length"
            )
            rows.append(payload)
        return _RunDataset(rows)


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> type[_ScriptedAnnotator]:
    _ScriptedAnnotator.new_columns = {}
    _ScriptedAnnotator.invalid_rows = set()
    _ScriptedAnnotator.num_calls = 0
    monkeypatch.setattr(dg, "set_seed", lambda seed: None)
    monkeypatch.setattr(dg, "Annotator", _ScriptedAnnotator)
    monkeypatch.setattr(
        dg,
        "_create_client_and_runtime_options",
        lambda step: (object(), object()),
    )
    monkeypatch.setattr(
        dg,
        "concatenate_datasets",
        lambda parts: _RunDataset([r for p in parts for r in p.rows]),
    )
    return _ScriptedAnnotator


def _patch_inputs(
    monkeypatch: pytest.MonkeyPatch, ds: _RunDataset, steps: list
) -> None:
    for idx, step in enumerate(steps):
        step.idx = idx
    monkeypatch.setattr(dg, "init_data", lambda **kwargs: ds)
    monkeypatch.setattr(
        dg, "build_pipeline", lambda config: _FakePipeline(steps)
    )


def test_run_generation_writes_invalid_samples_to_failed_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scripted: type[_ScriptedAnnotator],
) -> None:
    scripted.invalid_rows = {1}
    ds = _RunDataset([{"id": 0}, {"id": 1}, {"id": 2}])
    _patch_inputs(
        monkeypatch, ds, [_make_step(name="d", post_field_renamer={})]
    )

    dg.run_generation(
        dg.GenerateDefinitionsConfig(root_output_dir=tmp_path, hub_id=None)
    )

    step_dir = tmp_path / "1-definer-d"
    assert not (tmp_path / "failed_samples.jsonl").exists()
    lines = (step_dir / "failed_samples.jsonl").read_text().splitlines()
    failed = [json.loads(line) for line in lines]
    assert [f["id"] for f in failed] == [1]
    assert failed[0]["1-d_finish_reason"] == "length"
    assert failed[0]["invalid_reason"] == (
        "1-d: Did not produce valid_fields=True"
    )
    assert json.loads((step_dir / "attrition.json").read_text()) == {
        "n_in": 3,
        "n_valid": 2,
        "n_invalid": 1,
        "invalid_finish_reasons": {"length": 1},
    }


def test_failed_samples_file_is_overwritten_on_rerun(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scripted: type[_ScriptedAnnotator],
) -> None:
    step_dir = tmp_path / "1-definer-d"
    step_dir.mkdir()
    (step_dir / "failed_samples.jsonl").write_text('{"stale": 1}\n')
    _patch_inputs(
        monkeypatch,
        _RunDataset([{"id": 0}]),
        [_make_step(name="d", post_field_renamer={})],
    )

    dg.run_generation(
        dg.GenerateDefinitionsConfig(root_output_dir=tmp_path, hub_id=None)
    )

    assert (step_dir / "failed_samples.jsonl").read_text() == ""
    attrition = json.loads((step_dir / "attrition.json").read_text())
    assert attrition["n_invalid"] == 0


def test_run_generation_raises_when_all_samples_invalid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scripted: type[_ScriptedAnnotator],
) -> None:
    scripted.invalid_rows = {0}
    _patch_inputs(monkeypatch, _RunDataset([{"id": 0}]), [_make_step()])

    with pytest.raises(ValueError, match="All samples were filtered out"):
        dg.run_generation(
            dg.GenerateDefinitionsConfig(root_output_dir=tmp_path, hub_id=None)
        )


def test_run_generation_skips_step_with_existing_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scripted: type[_ScriptedAnnotator],
) -> None:
    done = tmp_path / "1-definer-d" / "done"
    done.mkdir(parents=True)
    (done / ".saved").write_text("1", encoding="utf-8")
    _patch_inputs(
        monkeypatch,
        _RunDataset([{"id": 0}]),
        [_make_step(name="d", post_field_renamer={})],
    )

    dg.run_generation(
        dg.GenerateDefinitionsConfig(root_output_dir=tmp_path, hub_id=None)
    )

    assert scripted.num_calls == 0
    assert (tmp_path / "full_process_done" / ".saved").exists()


def test_run_generation_overwrite_discards_existing_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scripted: type[_ScriptedAnnotator],
) -> None:
    done = tmp_path / "1-definer-d" / "done"
    done.mkdir(parents=True)
    (done / "stale.txt").write_text("old", encoding="utf-8")
    _patch_inputs(
        monkeypatch,
        _RunDataset([{"id": 0}]),
        [_make_step(name="d", post_field_renamer={})],
    )

    dg.run_generation(
        dg.GenerateDefinitionsConfig(
            root_output_dir=tmp_path, hub_id=None, overwrite=True
        )
    )

    assert scripted.num_calls == 1
    assert not (done / "stale.txt").exists()


REVISOR_SCHEMA = {
    "properties": {"samenvatting-feedback": {}, "nieuwe-definitie": {}}
}


def _revisor_pipeline() -> list:
    return [
        _make_step(name="d", post_field_renamer={}),
        _make_step(
            name="r",
            task="revisor",
            schema=REVISOR_SCHEMA,
            post_field_renamer={"nieuwe-definitie": "definitie-v2"},
        ),
    ]


def _revisor_rows() -> list[dict]:
    return [
        {"id": i, "definitie-v1": f"v1-{i}", "feedback": fb}
        for i, fb in enumerate(["fix", "", "fix", ""])
    ]


def _run_and_capture(
    monkeypatch: pytest.MonkeyPatch, out_dir: Path, steps: list
) -> list[dict]:
    pushed: list[_RunDataset] = []
    monkeypatch.setattr(
        _RunDataset, "push_to_hub", lambda self, *a, **k: pushed.append(self)
    )
    _patch_inputs(monkeypatch, _RunDataset(_revisor_rows()), steps)
    dg.run_generation(
        dg.GenerateDefinitionsConfig(root_output_dir=out_dir, hub_id="o/x")
    )
    return pushed[0].rows


def test_revisor_sends_only_rows_with_feedback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scripted: type[_ScriptedAnnotator],
) -> None:
    scripted.new_columns = {
        "nieuwe-definitie": "revised",
        "samenvatting-feedback": "sum",
    }
    sent: list[list[int]] = []
    original = scripted.annotate_dataset

    def spy(self, **kwargs):
        if kwargs["task_prefix"].startswith("2-"):
            sent.append(kwargs["dataset"]["id"])
        return original(self, **kwargs)

    monkeypatch.setattr(scripted, "annotate_dataset", spy)

    rows = _run_and_capture(monkeypatch, tmp_path, _revisor_pipeline())

    assert sent == [[0, 2]]
    assert [r["id"] for r in rows] == [0, 1, 2, 3]
    assert [r["revisie-uitgevoerd"] for r in rows] == [True, False] * 2
    assert [r["definitie-v2"] for r in rows] == [
        "revised",
        "v1-1",
        "revised",
        "v1-3",
    ]
    assert rows[1]["samenvatting-feedback"] is None
    assert rows[1]["2-r_valid_fields"] is True
    assert "__row_idx" not in rows[0]


def test_resumed_run_equals_fresh_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scripted: type[_ScriptedAnnotator],
) -> None:
    scripted.new_columns = {
        "nieuwe-definitie": "revised",
        "samenvatting-feedback": "sum",
    }
    fresh = _run_and_capture(monkeypatch, tmp_path, _revisor_pipeline())
    calls_after_fresh = scripted.num_calls

    resumed = _run_and_capture(monkeypatch, tmp_path, _revisor_pipeline())

    assert scripted.num_calls == calls_after_fresh
    assert resumed == fresh
    assert len(resumed) == 4


def test_revisor_without_feedback_skips_the_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scripted: type[_ScriptedAnnotator],
) -> None:
    rows = _revisor_rows()
    for row in rows:
        row["feedback"] = ""
    pushed: list[_RunDataset] = []
    monkeypatch.setattr(
        _RunDataset, "push_to_hub", lambda self, *a, **k: pushed.append(self)
    )
    _patch_inputs(monkeypatch, _RunDataset(rows), _revisor_pipeline())

    dg.run_generation(
        dg.GenerateDefinitionsConfig(root_output_dir=tmp_path, hub_id="o/x")
    )

    assert scripted.num_calls == 1
    assert [r["definitie-v2"] for r in pushed[0].rows] == [
        f"v1-{i}" for i in range(4)
    ]
    assert not any(r["revisie-uitgevoerd"] for r in pushed[0].rows)


def test_run_generation_splits_judge_scores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scripted: type[_ScriptedAnnotator],
) -> None:
    scripted.new_columns = {"raw": {"analyse": "goed", "score": 4}}
    pushed: list[_RunDataset] = []
    monkeypatch.setattr(
        _RunDataset, "push_to_hub", lambda self, *a, **k: pushed.append(self)
    )
    judged: list[list[str]] = []
    original = scripted.annotate_dataset

    def spy(self, **kwargs):
        if kwargs["task_prefix"].startswith("2-"):
            judged.append(kwargs["dataset"]["te-beoordelen-definitie"])
        return original(self, **kwargs)

    monkeypatch.setattr(scripted, "annotate_dataset", spy)
    steps = [
        _make_step(name="d", post_field_renamer={}),
        _make_step(
            name="j",
            task="judge",
            definition_column="originele-definitie",
            post_field_renamer={"raw": "begrijpelijkheid"},
        ),
    ]
    _patch_inputs(
        monkeypatch,
        _RunDataset([{"id": 0, "originele-definitie": "orig"}]),
        steps,
    )

    dg.run_generation(
        dg.GenerateDefinitionsConfig(root_output_dir=tmp_path, hub_id="o/x")
    )

    row = pushed[0].rows[0]
    assert row["analyse_begrijpelijkheid"] == "goed"
    assert row["score_begrijpelijkheid"] == 4
    assert "begrijpelijkheid" not in row
    assert judged == [["orig"]]
    assert "te-beoordelen-definitie" not in row


def test_run_generation_concats_feedback_after_last_feedback_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scripted: type[_ScriptedAnnotator],
) -> None:
    scripted.new_columns = {"raw_nodig": True, "raw_gebruik": "te vaag"}
    pushed: list[_RunDataset] = []
    monkeypatch.setattr(
        _RunDataset, "push_to_hub", lambda self, *a, **k: pushed.append(self)
    )
    steps = [
        _make_step(name="d", post_field_renamer={}),
        _make_step(
            name="f",
            task="feedback",
            post_field_renamer={
                "raw_nodig": "revisie-nodig_f",
                "raw_gebruik": "gebruik_f",
            },
        ),
    ]
    _patch_inputs(
        monkeypatch, _RunDataset([{"id": 0, "lemmavorm": "woord"}]), steps
    )

    dg.run_generation(
        dg.GenerateDefinitionsConfig(root_output_dir=tmp_path, hub_id="o/x")
    )

    assert pushed[0].rows[0]["feedback"] == (
        "*Feedback van reviewer 1*\n- gebruik: te vaag"
    )


@pytest.mark.parametrize(("num_samples", "expected"), [(2, 2), (None, 4)])
def test_run_generation_applies_num_samples_to_hub_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scripted: type[_ScriptedAnnotator],
    num_samples: int | None,
    expected: int,
) -> None:
    hub_ds = Dataset.from_list([{"id": i} for i in range(4)])
    monkeypatch.setattr(dg, "load_dataset", lambda hub_id: {"train": hub_ds})
    seen: list[int] = []

    def annotate(self, **kwargs):
        seen.append(len(kwargs["dataset"]))
        prefix = kwargs["task_prefix"]
        return _RunDataset(
            [
                {
                    **row,
                    f"{prefix}valid_fields": True,
                    f"{prefix}finish_reason": "stop",
                }
                for row in kwargs["dataset"].to_list()
            ]
        )

    monkeypatch.setattr(scripted, "annotate_dataset", annotate)
    monkeypatch.setattr(
        dg,
        "build_pipeline",
        lambda config: _FakePipeline(
            [_make_step(name="d", post_field_renamer={})]
        ),
    )

    dg.run_generation(
        dg.GenerateDefinitionsConfig(
            root_output_dir=tmp_path,
            hub_id=None,
            from_hub_id="o/hub",
            num_samples=num_samples,
        )
    )

    assert seen == [expected]


def _candidate_ds() -> Dataset:
    ds = Dataset.from_list(
        [
            {
                "definitie-v1": "v1",
                "definitie-v2": "v2",
                "originele-definitie": "orig",
                "lemmavorm": f"woord{i}",
            }
            for i in range(5)
        ]
    )
    return ds.map(dg.shuffle_definitions)


def test_ranker_rotation_zero_matches_shuffled_definitions() -> None:
    ds = _candidate_ds()
    step = _make_step(name="rk", task="ranker")

    out = dg._add_ranker_positions(ds, step, num_proc=1)

    for row in out:
        for i in (1, 2, 3):
            assert (
                row[f"positie-definitie-{i}"] == row[f"random-definitie-{i}"]
            )
        assert row["volgorde_rk"] == row["shuffle-order"]


def test_ranker_rotation_shifts_candidates_left() -> None:
    ds = _candidate_ds()
    step = _make_step(name="rk", task="ranker", order_rotation=1)

    out = dg._add_ranker_positions(ds, step, num_proc=1)

    for row in out:
        assert row["volgorde_rk"] == [
            row["shuffle-order"][1],
            row["shuffle-order"][2],
            row["shuffle-order"][0],
        ]
        assert row["positie-definitie-1"] == row["random-definitie-2"]
        assert row["positie-definitie-3"] == row["random-definitie-1"]


def test_ranker_rotation_must_be_below_candidate_count() -> None:
    step = _make_step(name="rk", task="ranker", order_rotation=3)

    with pytest.raises(ValueError, match="order_rotation 3"):
        dg._add_ranker_positions(_candidate_ds(), step, num_proc=1)


def test_run_generation_drops_position_columns_and_keeps_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scripted: type[_ScriptedAnnotator],
) -> None:
    pushed: list[_RunDataset] = []
    monkeypatch.setattr(
        _RunDataset, "push_to_hub", lambda self, *a, **k: pushed.append(self)
    )
    row = {"definitie-v1": "a", "originele-definitie": "b"}
    steps = [
        _make_step(name="d", post_field_renamer={}),
        _make_step(
            name="rk", task="ranker", order_rotation=1, post_field_renamer={}
        ),
    ]
    _patch_inputs(monkeypatch, _RunDataset([row]), steps)

    dg.run_generation(
        dg.GenerateDefinitionsConfig(root_output_dir=tmp_path, hub_id="o/x")
    )

    out = pushed[0].rows[0]
    assert out["volgorde_rk"] == out["shuffle-order"][::-1]
    assert not any(c.startswith("positie-definitie-") for c in out)
