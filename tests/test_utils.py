import json
from pathlib import Path

import pytest

from def_gen import utils
from def_gen.utils import PathCompatibleEncoder, run_metadata, safe_json_loads


def test_safe_json_loads_parses_valid_json() -> None:
    assert safe_json_loads('{"a": [1, 2]}') == {"a": [1, 2]}


def test_safe_json_loads_reports_position_and_snippet() -> None:
    with pytest.raises(ValueError, match=r"pos 6.*snippet='\{\"a\": oops\}'"):
        safe_json_loads('{"a": oops}')


def test_encoder_converts_paths_and_rejects_other_types() -> None:
    assert json.dumps({"p": Path("a/b")}, cls=PathCompatibleEncoder) == (
        '{"p": "a/b"}'
    )
    with pytest.raises(TypeError):
        json.dumps({"s": {1}}, cls=PathCompatibleEncoder)


def test_run_metadata_reports_versions_and_timestamp() -> None:
    meta = run_metadata()

    assert set(meta["versions"]) == {
        "def-gen",
        "llm-annotator",
        "datasets",
        "vllm",
    }
    assert meta["versions"]["datasets"] is not None
    assert meta["timestamp_utc"].endswith("+00:00")
    assert len(meta["git_commit"]) == 40


def test_run_metadata_without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(utils.subprocess, "run", no_git)

    assert run_metadata()["git_commit"] is None
