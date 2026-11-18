"""Merge the definitions of several writer runs into one dataset for ranking.

Example:
    uv run python scripts/merge_writer_outputs.py \
        outputs/study/writer-gemma4-26b-a4b \
        outputs/study/writer-qwen36-35b-a3b \
        --output_dir outputs/study/head-to-head-input
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk


ID_COL = "betekenis_pid"
SHARED_COLS = [ID_COL, "lemmavorm", "voorbeelden", "originele-definitie"]


def writer_name(run_dir: Path) -> str:
    """Return the writer name from the definer step of `pipeline.json`.

    Example: the step `definer-qwen36-35b-a3b` gives `qwen36-35b-a3b`.

    Args:
        run_dir: Output directory of a writer run.

    Returns:
        The definer step name without the `definer-` prefix.
    """
    steps = json.loads(
        run_dir.joinpath("pipeline.json").read_text(encoding="utf-8")
    )["steps"]
    definers = [s["name"] for s in steps if s["task"] == "definer"]
    assert len(definers) == 1, f"{run_dir} needs exactly one definer step"
    return definers[0].removeprefix("definer-")


def main(run_dirs: list[str], output_dir: str) -> None:
    """Join the `definitie-v1` columns of writer runs on `betekenis_pid`.

    Only items that are present in every run are kept. The output has the
    columns `betekenis_pid`, `lemmavorm`, `voorbeelden`,
    `originele-definitie` and one `definitie-<writer>` per run, in the split
    `eval`.

    Args:
        run_dirs: Output directories of two or more writer runs.
        output_dir: Directory that the DatasetDict is saved to.
    """
    assert len(run_dirs) >= 2, "Need at least two writer runs"
    runs = {}
    for run_dir in map(Path, run_dirs):
        name = writer_name(run_dir)
        assert name not in runs, f"Duplicate writer {name!r}"
        ds = load_from_disk(str(run_dir / "full_process_done"))
        ids = ds[ID_COL]
        assert len(ids) == len(set(ids)), f"{name}: {ID_COL} is not unique"
        runs[name] = ds

    common = set.intersection(*(set(ds[ID_COL]) for ds in runs.values()))
    for name, ds in runs.items():
        print(f"{name}: dropped {len(ds) - len(common)} of {len(ds)} items")

    rows = {}
    for name, ds in runs.items():
        for row in ds:
            if row[ID_COL] not in common:
                continue
            merged = rows.setdefault(
                row[ID_COL], {c: row[c] for c in SHARED_COLS}
            )
            merged[f"definitie-{name}"] = row["definitie-v1"]

    assert len(rows) == len(common)
    merged_ds = Dataset.from_list(list(rows.values()))
    DatasetDict({"eval": merged_ds}).save_to_disk(output_dir)
    print(f"Saved {len(merged_ds)} items to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--output_dir", required=True)
    main(**vars(parser.parse_args()))
