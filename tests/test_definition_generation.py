from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from def_gen import definition_generation as dg


class _FakeTorch:
    class cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def manual_seed_all(seed: int) -> None:
            _FakeTorch.cuda_seed = seed

    manual_seed_value = None
    cuda_seed = None

    @staticmethod
    def manual_seed(seed: int) -> None:
        _FakeTorch.manual_seed_value = seed


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
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_config_coerces_path_and_defaults() -> None:
    config = dg.GenerateDefinitionsConfig(root_output_dir="outputs/custom")

    assert config.root_output_dir == Path("outputs/custom")
    assert config.workflow == dg.WORKFLOW_NO_REVISION
    assert config.add_revision_stage is False
    assert config.batch_size == 256


def test_config_rejects_removed_or_unknown_values() -> None:
    with pytest.raises(ValidationError):
        dg.GenerateDefinitionsConfig(gpu_memory_utilization=1.5)


def test_from_sources_cli_overrides_config_file() -> None:
    config = dg.GenerateDefinitionsConfig.from_sources(
        config_data={"num_samples": 10, "workflow": "no-revision"},
        overrides={"num_samples": 2, "workflow": "with-revision"},
    )

    assert config.num_samples == 2
    assert config.workflow == dg.WORKFLOW_WITH_REVISION


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
            "client_run_kwargs": {"max_tokens": 12},
            "schema": {"type": "object"},
        }
    )

    out = step.to_step()

    assert step.output_schema == {"type": "object"}
    assert out.provider == "openai"
    assert out.client_init_kwargs["api_key"] == "x"
    assert out.client_run_kwargs["max_tokens"] == 12


def test_set_seed_calls_torch_and_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"np_seed": None}

    monkeypatch.setattr(
        dg.np.random,
        "seed",
        lambda seed: calls.__setitem__("np_seed", seed),
    )
    monkeypatch.setattr(dg, "_load_torch", lambda: _FakeTorch)

    dg.set_seed(123)

    assert calls == {"np_seed": 123}
    assert _FakeTorch.manual_seed_value == 123
    assert _FakeTorch.cuda_seed == 123


def test_set_seed_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"np_seed": None}
    monkeypatch.setattr(
        dg.np.random,
        "seed",
        lambda seed: calls.__setitem__("np_seed", seed),
    )
    monkeypatch.setattr(dg, "_load_torch", lambda: None)

    dg.set_seed(7)

    assert calls == {"np_seed": 7}


def test_resolve_provider_uses_explicit_value() -> None:
    explicit = _make_step(
        model_name="claude-sonnet-4",
        provider="anthropic",
    )

    assert dg._resolve_provider(explicit) == "anthropic"


def test_resolve_provider_vllm_server_and_offline() -> None:
    server = _make_step(client_init_kwargs={"base_url": "http://localhost:8000/v1"})
    offline = _make_step()

    assert dg._resolve_provider(server) == "vllm"
    assert dg._resolve_provider(offline) == "vllm-offline"


def test_resolve_provider_rejects_invalid_value() -> None:
    step = _make_step(provider="unknown")

    with pytest.raises(ValueError, match="provider"):
        dg._resolve_provider(step)


def test_create_client_uses_expected_client_class() -> None:
    step = _make_step(
        model_name="gpt-4o-mini",
        provider="openai",
    )

    client = dg._create_client(step)

    assert client.model == "gpt-4o-mini"


def test_build_runtime_options_uses_client_run_kwargs() -> None:
    step = _make_step(
        model_name="gpt-4o-mini",
        provider="openai",
        client_run_kwargs={"max_tokens": 333, "temperature": 0.2},
    )

    options = dg._build_runtime_options(step)

    assert options.kwargs["max_tokens"] == 333
    assert options.kwargs["temperature"] == 0.2


def test_concat_feedback_skips_repeated_characters() -> None:
    sample = {
        "revisie-nodig_a": True,
        "gebruik_a": "goede feedback",
        "structuur_a": "aaaaa",
        "lemmavorm": "woord",
    }
    feedback_cols = {"feedback-a": ["revisie-nodig_a", "gebruik_a", "structuur_a"]}

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
    sample_bad = {"c1": "Definitie 1", "c2": "Definitie 1", "c3": "Definitie 3"}

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
                client_run_kwargs={"max_tokens": 42},
                post_field_renamer={"definitie": "definitie-v1"},
            )
        ]
    )

    pipe = dg.build_pipeline(config)

    assert [step.name for step in pipe] == ["custom-definer"]
    assert pipe[0].provider == "openai"
    assert pipe[0].client_run_kwargs["max_tokens"] == 42


class _RunDataset:
    pushed = None

    def __init__(self, rows):
        self.rows = rows

    @property
    def column_names(self):
        return list(self.rows[0].keys()) if self.rows else []

    def load_from_disk(self, path):
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
        for row, value in zip(self.rows, values):
            payload = dict(row)
            payload[col] = value
            rows.append(payload)
        return _RunDataset(rows)

    def to_list(self):
        return [dict(row) for row in self.rows]

    def save_to_disk(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        path.joinpath(".saved").write_text("1", encoding="utf-8")

    def push_to_hub(self, hub_id, private=True):
        _RunDataset.pushed = (hub_id, private)

    def __getitem__(self, key):
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
        return {"steps": [{"name": step.name, "task": step.task} for step in self.steps]}


class _FakeAnnotator:
    init_calls = []

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


def test_run_generation_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _FakeAnnotator.init_calls = []
    base_ds = _RunDataset([{"definitie": "Een voorbeeld definitie", "feedback": ""}])

    fake_step = _make_step(name="definer-fake")

    monkeypatch.setattr(dg, "set_seed", lambda seed: None)
    monkeypatch.setattr(dg, "init_data", lambda **kwargs: base_ds)
    monkeypatch.setattr(dg, "build_pipeline", lambda config: _FakePipeline([fake_step]))
    monkeypatch.setattr(dg, "Annotator", _FakeAnnotator)
    monkeypatch.setattr(dg, "_create_client", lambda step: object())
    monkeypatch.setattr(dg, "_build_runtime_options", lambda step: object())

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
    monkeypatch.setattr(dg, "build_pipeline", lambda config: _FakePipeline([fake_step]))
    monkeypatch.setattr(dg, "Annotator", _FakeAnnotator)
    monkeypatch.setattr(dg, "_create_client", lambda step: object())
    monkeypatch.setattr(dg, "_build_runtime_options", lambda step: object())

    config = dg.GenerateDefinitionsConfig(
        root_output_dir=tmp_path,
        hub_id="owner/ds-{num_samples}",
    )
    dg.run_generation(config)

    assert _RunDataset.pushed == ("owner/ds-1", True)
