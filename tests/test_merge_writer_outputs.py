import importlib.util
import json
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict, load_from_disk


_SPEC = importlib.util.spec_from_file_location(
    "merge_writer_outputs",
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "merge_writer_outputs.py",
)
assert _SPEC and _SPEC.loader is not None
merge = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(merge)


def _make_run(root: Path, writer: str, pids: list[str]) -> Path:
    run = root / writer
    rows = [
        {
            "betekenis_pid": p,
            "lemmavorm": f"lemma-{p}",
            "voorbeelden": f"vb-{p}",
            "originele-definitie": f"orig-{p}",
            "definitie-v1": f"{writer}-{p}",
        }
        for p in pids
    ]
    Dataset.from_list(rows).save_to_disk(run / "full_process_done")
    steps = [
        {"name": f"definer-{writer}", "task": "definer"},
        {"name": "judge-x", "task": "judge"},
    ]
    run.joinpath("pipeline.json").write_text(json.dumps({"steps": steps}))
    return run


def test_writer_name(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "qwen36-35b-a3b", ["a"])

    assert merge.writer_name(run) == "qwen36-35b-a3b"


def test_merge_keeps_common_items_and_reports_drops(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_a = _make_run(tmp_path, "wa", ["p1", "p2", "p3"])
    run_b = _make_run(tmp_path, "wb", ["p3", "p1", "p4"])
    out = tmp_path / "merged"

    merge.main([str(run_a), str(run_b)], str(out))

    ds = load_from_disk(str(out))
    assert isinstance(ds, DatasetDict)
    rows = {r["betekenis_pid"]: r for r in ds["eval"]}
    assert set(rows) == {"p1", "p3"}
    assert rows["p1"]["definitie-wa"] == "wa-p1"
    assert rows["p1"]["definitie-wb"] == "wb-p1"
    assert rows["p3"]["originele-definitie"] == "orig-p3"
    assert set(ds["eval"].column_names) == {
        "betekenis_pid",
        "lemmavorm",
        "voorbeelden",
        "originele-definitie",
        "definitie-wa",
        "definitie-wb",
    }
    printed = capsys.readouterr().out
    assert "wa: dropped 1 of 3" in printed
    assert "wb: dropped 1 of 3" in printed


def test_merge_rejects_duplicate_ids_and_single_run(tmp_path: Path) -> None:
    dup = _make_run(tmp_path, "wa", ["p1", "p1"])
    other = _make_run(tmp_path, "wb", ["p1"])

    with pytest.raises(AssertionError, match="not unique"):
        merge.main([str(dup), str(other)], str(tmp_path / "out"))
    with pytest.raises(AssertionError, match="at least two"):
        merge.main([str(other)], str(tmp_path / "out"))
