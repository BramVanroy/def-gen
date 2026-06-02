import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "generate_definitions_script",
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "generate_definitions.py",
)
assert _SPEC and _SPEC.loader is not None
cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cli)


def test_load_json_config_valid_and_invalid(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"num_samples": 3}), encoding="utf-8")

    assert cli.load_json_config(config_path)["num_samples"] == 3

    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError):
        cli.load_json_config(bad_path)


def test_parse_args_requires_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["generate_definitions.py", "cfg.json"])

    args = cli.parse_args()

    assert args.config == Path("cfg.json")


def test_parse_args_missing_config_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["generate_definitions.py"])

    with pytest.raises(SystemExit):
        cli.parse_args()
