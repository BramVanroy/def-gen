"""Read pipeline runs into long tables for the analysis.

A run directory holds `pipeline.json`, the final dataset in
`full_process_done` and an `attrition.json` per step directory. One run has
at most one writer (its definer step). The ANW definitions are the system
"anw" and every other system is named after the short model name of its
writer.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_from_disk


ITEM_COLUMN = "betekenis_pid"
ANW_COLUMN = "originele-definitie"
WRITER_COLUMN = "definitie-v1"
ANW_SYSTEM = "anw"
RANKING_LABELS = {"1 beter dan 2": 1, "2 beter dan 1": 2, "evenwaardig": None}
MODEL_FAMILIES = ("gpt-oss", "qwen", "gemma")

SCORE_KEYS = ["item", "system", "judge", "criterion"]
PAIR_KEYS = ["item", "judge", "rotation", "system_a", "system_b"]


@dataclass
class AnalysisData:
    """Long tables of one or more runs.

    Args:
        scores: One row per item, system, judge and criterion with the
            columns run, item, system, judge, criterion and score.
        pairs: One row per pairwise judgment with the columns run, item,
            judge, rotation, system_a, system_b, position_a (1 or 2, where
            system_a was shown), outcome ("a", "b" or "tie") and
            winner_position (1, 2 or None for a tie). If one of the systems
            is "anw", it is system_b.
        definitions: One row per item and system with the definition text.
        attrition: One row per step with run, step, task, model, n_in,
            n_valid, n_invalid and invalid_finish_reasons (a JSON string).
        coverage: One row per system and judge (scores) or system pair,
            judge and rotation (pairs) with the number of items before and
            after the restriction to the common items.
    """

    scores: pd.DataFrame
    pairs: pd.DataFrame
    definitions: pd.DataFrame
    attrition: pd.DataFrame
    coverage: pd.DataFrame = field(default_factory=pd.DataFrame)


def short_model_name(model_name: str) -> str:
    """Return the model name without its organisation prefix.

    Args:
        model_name: A model identifier such as "Qwen/Qwen3.6-27B-FP8".

    Returns:
        The part after the last slash.

    Examples:
        >>> short_model_name("Qwen/Qwen3.6-27B-FP8")
        'Qwen3.6-27B-FP8'
    """
    return model_name.rstrip("/").split("/")[-1]


def model_family(model_name: str) -> str:
    """Return the model family (gpt-oss, qwen or gemma) of a model name.

    Args:
        model_name: A full or short model name.

    Returns:
        The first known family that occurs in the lowercased name, or the
        lowercased short name if none does.

    Examples:
        >>> model_family("RedHatAI/gemma-3-27b-it-FP8-dynamic")
        'gemma'
        >>> model_family("openai/gpt-oss-120b")
        'gpt-oss'
    """
    name = short_model_name(model_name).lower()
    return next((fam for fam in MODEL_FAMILIES if fam in name), name)


def _system_for_column(column: str, writer: str | None) -> str:
    """Return the system name of a definition column in a run."""
    if column == ANW_COLUMN:
        return ANW_SYSTEM
    if writer is None:
        raise ValueError(
            f"Column {column!r} is not the ANW definition, but the run has"
            " no definer step to name its writer."
        )
    if column == WRITER_COLUMN:
        return writer
    return f"{writer}/{column}"


def _run_writer(steps: list[dict[str, Any]]) -> str | None:
    """Return the short model name of the single definer step, if any."""
    definers = [s for s in steps if s["task"] == "definer"]
    if len(definers) > 1:
        raise ValueError("A run with more than one definer is not supported.")
    return short_model_name(definers[0]["model_name"]) if definers else None


def _require_columns(df: pd.DataFrame, columns: list[str], step: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Step {step!r} lacks the columns {missing}.")


def scores_from_run(
    df: pd.DataFrame, steps: list[dict[str, Any]], run: str
) -> pd.DataFrame:
    """Collect the judge scores of one run in long format.

    Args:
        df: The final dataset of the run as a DataFrame.
        steps: The steps from `pipeline.json`.
        run: Name of the run, stored in the "run" column.

    Returns:
        A DataFrame with the columns run, item, system, judge, criterion and
        score.
    """
    writer = _run_writer(steps)
    frames = []
    for step in steps:
        if step["task"] != "judge":
            continue
        system = _system_for_column(step["definition_column"], writer)
        renamer = step["post_field_renamer"]
        _require_columns(
            df, [f"score_{r}" for r in renamer.values()], step["name"]
        )
        for criterion, renamed in renamer.items():
            frames.append(
                pd.DataFrame(
                    {
                        "run": run,
                        "item": df[ITEM_COLUMN],
                        "system": system,
                        "judge": short_model_name(step["model_name"]),
                        "criterion": criterion,
                        "score": df[f"score_{renamed}"],
                    }
                )
            )
    columns = ["run", *SCORE_KEYS, "score"]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


def pairs_from_run(
    df: pd.DataFrame, steps: list[dict[str, Any]], run: str
) -> pd.DataFrame:
    """Collect the two-way ranker judgments of one run in long format.

    The ranker answers "1 beter dan 2", "2 beter dan 1" or "evenwaardig" about
    the definitions at position 1 and 2. `volgorde_<step name>` holds the
    source columns in position order. The outcome is expressed relative to
    the canonical pair, in which "anw" is always system_b.

    Args:
        df: The final dataset of the run as a DataFrame.
        steps: The steps from `pipeline.json`.
        run: Name of the run, stored in the "run" column.

    Returns:
        A DataFrame with the columns of `AnalysisData.pairs`.

    Raises:
        ValueError: If a ranker step compares more than two definitions or
            gives an unknown label.

    Examples:
        If the order is ["originele-definitie", "definitie-v1"] and the
        ranker says "2 beter dan 1", the writer (system_a, shown at position
        2) wins: position_a is 2, outcome is "a" and winner_position is 2.
    """
    writer = _run_writer(steps)
    rows = []
    for step in steps:
        if step["task"] != "ranker":
            continue
        ranking_col = step["post_field_renamer"].get("ranking", "ranking")
        order_col = f"volgorde_{step['name']}"
        _require_columns(df, [ranking_col, order_col], step["name"])
        judge = short_model_name(step["model_name"])
        for item, order, label in zip(
            df[ITEM_COLUMN], df[order_col], df[ranking_col], strict=True
        ):
            if len(order) != 2:
                raise ValueError(
                    f"Step {step['name']!r} ranks {len(order)} definitions;"
                    " only two-way rankings are supported."
                )
            if label not in RANKING_LABELS:
                raise ValueError(
                    f"Unknown ranking {label!r} in step {step['name']!r}."
                )
            shown = [_system_for_column(c, writer) for c in order]
            system_a, system_b = sorted(
                shown, key=lambda s: (s == ANW_SYSTEM, s)
            )
            position_a = shown.index(system_a) + 1
            winner = RANKING_LABELS[label]
            if winner is None:
                outcome = "tie"
            else:
                outcome = "a" if winner == position_a else "b"
            rows.append(
                {
                    "run": run,
                    "item": item,
                    "judge": judge,
                    "rotation": step["order_rotation"],
                    "system_a": system_a,
                    "system_b": system_b,
                    "position_a": position_a,
                    "outcome": outcome,
                    "winner_position": winner,
                }
            )
    columns = [
        "run",
        *PAIR_KEYS,
        "position_a",
        "outcome",
        "winner_position",
    ]
    return pd.DataFrame(rows, columns=columns)


def definitions_from_run(
    df: pd.DataFrame, steps: list[dict[str, Any]]
) -> pd.DataFrame:
    """Collect the ANW and writer definitions of one run.

    Args:
        df: The final dataset of the run as a DataFrame.
        steps: The steps from `pipeline.json`.

    Returns:
        A DataFrame with the columns item, system and definition.
    """
    writer = _run_writer(steps)
    sources = {ANW_COLUMN: ANW_SYSTEM}
    if writer is not None:
        sources[WRITER_COLUMN] = writer
    frames = [
        pd.DataFrame(
            {"item": df[ITEM_COLUMN], "system": system, "definition": df[col]}
        )
        for col, system in sources.items()
        if col in df.columns
    ]
    return pd.concat(frames, ignore_index=True)


def attrition_from_run(
    run_dir: Path, steps: list[dict[str, Any]]
) -> pd.DataFrame:
    """Read the `attrition.json` of every step of a run.

    A step without the file (for example because it received no rows) gets
    missing counts.

    Args:
        run_dir: The run directory.
        steps: The steps from `pipeline.json`.

    Returns:
        A DataFrame with the columns of `AnalysisData.attrition`.
    """
    rows = []
    for step in steps:
        step_dir = run_dir / f"{step['idx'] + 1}-{step['task']}-{step['name']}"
        path = step_dir / "attrition.json"
        counts = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.is_file()
            else {}
        )
        rows.append(
            {
                "run": run_dir.name,
                "step": step["name"],
                "task": step["task"],
                "model": short_model_name(step["model_name"]),
                "n_in": counts.get("n_in"),
                "n_valid": counts.get("n_valid"),
                "n_invalid": counts.get("n_invalid"),
                "invalid_finish_reasons": json.dumps(
                    counts.get("invalid_finish_reasons", {})
                ),
            }
        )
    return pd.DataFrame(rows)


def read_run(run_dir: str | Path) -> AnalysisData:
    """Read one run directory into long tables.

    Args:
        run_dir: Directory with `pipeline.json` and `full_process_done`.

    Returns:
        The tables of the run, with an empty coverage table.
    """
    run_dir = Path(run_dir)
    steps = json.loads(
        run_dir.joinpath("pipeline.json").read_text(encoding="utf-8")
    )["steps"]
    df = load_from_disk(run_dir / "full_process_done").to_pandas()
    return AnalysisData(
        scores=scores_from_run(df, steps, run_dir.name),
        pairs=pairs_from_run(df, steps, run_dir.name),
        definitions=definitions_from_run(df, steps),
        attrition=attrition_from_run(run_dir, steps),
    )


def _check_unique(df: pd.DataFrame, keys: list[str], name: str) -> None:
    duplicated = df.duplicated(keys, keep=False)
    if duplicated.any():
        example = df.loc[duplicated, keys].iloc[0].to_dict()
        raise ValueError(
            f"{int(duplicated.sum())} {name} rows share their keys across"
            f" runs, for example {example}."
        )


def combine_runs(runs: list[AnalysisData]) -> AnalysisData:
    """Combine runs and keep only the items that every comparison has.

    Each score group (system, judge) and pair group (system_a, system_b,
    judge, rotation) defines a set of items. Only the intersection of all
    these sets is kept, so every comparison uses the same items. The
    coverage table records how many items each group lost.

    Args:
        runs: Tables from `read_run`, for example two writer runs and one
            run in which only the ANW definitions are judged.

    Returns:
        The combined tables restricted to the common items.

    Raises:
        ValueError: If two runs hold the same score or pair, if the same item
            and system have different definition texts, or if no item is
            common to all groups.
    """
    scores = pd.concat([r.scores for r in runs], ignore_index=True)
    pairs = pd.concat([r.pairs for r in runs], ignore_index=True)
    definitions = pd.concat(
        [r.definitions for r in runs], ignore_index=True
    ).drop_duplicates()
    attrition = pd.concat([r.attrition for r in runs], ignore_index=True)
    _check_unique(scores, SCORE_KEYS, "score")
    _check_unique(pairs, PAIR_KEYS, "pair")
    _check_unique(definitions, ["item", "system"], "definition")

    groups = [
        ("scores", keys, set(group["item"]))
        for keys, group in scores.groupby(["system", "judge"])
    ]
    groups += [
        ("pairs", keys, set(group["item"]))
        for keys, group in pairs.groupby(
            ["system_a", "system_b", "judge", "rotation"]
        )
    ]
    if not groups:
        raise ValueError("The runs contain no scores and no pairs.")
    common = set.intersection(*(items for _, _, items in groups))
    if not common:
        raise ValueError("No item is present in every comparison.")

    coverage = pd.DataFrame(
        [
            {
                "table": table,
                "group": "/".join(str(k) for k in keys),
                "n_items": len(items),
                "n_kept": len(items & common),
                "n_dropped": len(items - common),
            }
            for table, keys, items in groups
        ]
    )
    return AnalysisData(
        scores=scores[scores["item"].isin(common)].reset_index(drop=True),
        pairs=pairs[pairs["item"].isin(common)].reset_index(drop=True),
        definitions=definitions[definitions["item"].isin(common)].reset_index(
            drop=True
        ),
        attrition=attrition,
        coverage=coverage,
    )
