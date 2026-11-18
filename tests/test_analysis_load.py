import json

import pandas as pd
import pytest

from def_gen.analysis.load import (
    AnalysisData,
    combine_runs,
    definitions_from_run,
    model_family,
    pairs_from_run,
    read_run,
    scores_from_run,
)
from def_gen.definition_generation import JUDGE_FIELDS
from tests.conftest import make_step, write_run


GEMMA = "RedHatAI/gemma-3-27b-it-FP8-dynamic"
QWEN = "Qwen/Qwen3.6-27B-FP8"


def _score_columns(step_name: str, values: list[int]) -> dict[str, list]:
    columns = {}
    for field in JUDGE_FIELDS:
        columns[f"score_{field}_{step_name}"] = values
        columns[f"analyse_{field}_{step_name}"] = ["ok"] * len(values)
    return columns


def _writer_run(tmp_path, name="writer-gemma", items=(1, 2, 3)):
    items = list(items)
    steps = [
        make_step(0, "definer", "definer", GEMMA),
        make_step(
            1, "judge-q", "judge", QWEN, definition_column="definitie-v1"
        ),
        make_step(
            2,
            "rank-q0",
            "ranker",
            QWEN,
            post_field_renamer={"ranking": "ranking_q0"},
        ),
        make_step(
            3,
            "rank-q1",
            "ranker",
            QWEN,
            order_rotation=1,
            post_field_renamer={"ranking": "ranking_q1"},
        ),
    ]
    order = ["originele-definitie", "definitie-v1"]
    columns = {
        "betekenis_pid": items,
        "originele-definitie": [f"anw {i}" for i in items],
        "definitie-v1": [f"gemma {i}" for i in items],
        **_score_columns("judge-q", [4, 5, 3][: len(items)]),
        "volgorde_rank-q0": [order] * len(items),
        "volgorde_rank-q1": [order[::-1]] * len(items),
        "ranking_q0": ["2 beter dan 1", "1 beter dan 2", "evenwaardig"][
            : len(items)
        ],
        "ranking_q1": ["1 beter dan 2", "1 beter dan 2", "evenwaardig"][
            : len(items)
        ],
    }
    attrition = {
        "judge-q": {
            "n_in": 4,
            "n_valid": 3,
            "n_invalid": 1,
            "invalid_finish_reasons": {"length": 1},
        }
    }
    return write_run(tmp_path / name, steps, columns, attrition)


def _anw_run(tmp_path, items=(1, 2, 3, 4)):
    items = list(items)
    steps = [
        make_step(
            0,
            "judge-q",
            "judge",
            QWEN,
            definition_column="originele-definitie",
        )
    ]
    columns = {
        "betekenis_pid": items,
        "originele-definitie": [f"anw {i}" for i in items],
        **_score_columns("judge-q", [2] * len(items)),
    }
    return write_run(tmp_path / "anw", steps, columns)


def test_model_family_falls_back_to_name() -> None:
    assert model_family("Qwen/Qwen3.6-27B-FP8") == "qwen"
    assert model_family("org/Llama-3") == "llama-3"


def test_read_run_scores(tmp_path) -> None:
    data = read_run(_writer_run(tmp_path))

    assert len(data.scores) == 3 * len(JUDGE_FIELDS)
    assert set(data.scores["system"]) == {"gemma-3-27b-it-FP8-dynamic"}
    assert set(data.scores["judge"]) == {"Qwen3.6-27B-FP8"}
    first = data.scores[data.scores["criterion"] == "betekenis"]
    assert first["score"].tolist() == [4, 5, 3]


def test_read_run_pairs_are_canonical(tmp_path) -> None:
    pairs = read_run(_writer_run(tmp_path)).pairs

    assert set(pairs["system_b"]) == {"anw"}
    rot0 = pairs[pairs["rotation"] == 0].set_index("item")
    rot1 = pairs[pairs["rotation"] == 1].set_index("item")
    # Rotation 0 shows ANW first, so the writer is at position 2.
    assert rot0["position_a"].tolist() == [2, 2, 2]
    assert rot0["outcome"].tolist() == ["a", "b", "tie"]
    assert rot0.loc[1, "winner_position"] == 2
    assert pd.isna(rot0.loc[3, "winner_position"])
    assert rot1["position_a"].tolist() == [1, 1, 1]
    assert rot1["outcome"].tolist() == ["a", "a", "tie"]


def test_read_run_definitions_and_attrition(tmp_path) -> None:
    data = read_run(_writer_run(tmp_path))

    assert len(data.definitions) == 6
    attrition = data.attrition.set_index("step")
    assert attrition.loc["judge-q", "n_invalid"] == 1
    assert json.loads(attrition.loc["judge-q", "invalid_finish_reasons"]) == {
        "length": 1
    }
    assert pd.isna(attrition.loc["definer", "n_in"])


def test_combine_runs_keeps_common_items(tmp_path) -> None:
    writer = read_run(_writer_run(tmp_path))
    anw = read_run(_anw_run(tmp_path))

    data = combine_runs([writer, anw])

    assert set(data.scores["item"]) == {1, 2, 3}
    assert set(data.scores["system"]) == {"anw", "gemma-3-27b-it-FP8-dynamic"}
    anw_cov = data.coverage[data.coverage["group"] == "anw/Qwen3.6-27B-FP8"]
    assert anw_cov[["n_items", "n_kept", "n_dropped"]].iloc[0].tolist() == [
        4,
        3,
        1,
    ]
    assert len(data.definitions) == 6


def test_combine_runs_rejects_duplicate_scores(tmp_path) -> None:
    anw = read_run(_anw_run(tmp_path))

    with pytest.raises(ValueError, match="share their keys"):
        combine_runs([anw, anw])


def test_combine_runs_without_common_items(tmp_path) -> None:
    writer = read_run(_writer_run(tmp_path, items=(1, 2)))
    anw = read_run(_anw_run(tmp_path, items=(7, 8)))

    with pytest.raises(ValueError, match="No item"):
        combine_runs([writer, anw])


def test_combine_runs_without_tables() -> None:
    steps = [make_step(0, "d", "definer", GEMMA)]
    df = pd.DataFrame({"betekenis_pid": [1], "definitie-v1": ["x"]})
    data = AnalysisData(
        scores=scores_from_run(df, steps, "run"),
        pairs=pairs_from_run(df, steps, "run"),
        definitions=definitions_from_run(df, steps),
        attrition=pd.DataFrame(),
    )

    with pytest.raises(ValueError, match="no scores and no pairs"):
        combine_runs([data])


def test_pairs_reject_three_way_and_unknown_labels() -> None:
    steps = [
        make_step(0, "definer", "definer", GEMMA),
        make_step(1, "rank", "ranker", QWEN),
    ]
    df = pd.DataFrame(
        {
            "betekenis_pid": [1],
            "volgorde_rank": [["originele-definitie", "definitie-v1"]],
            "ranking": ["beter"],
        }
    )
    with pytest.raises(ValueError, match="Unknown ranking"):
        pairs_from_run(df, steps, "run")

    df["volgorde_rank"] = [["a", "b", "c"]]
    with pytest.raises(ValueError, match="two-way"):
        pairs_from_run(df, steps, "run")


def test_scores_require_writer_and_columns() -> None:
    df = pd.DataFrame({"betekenis_pid": [1]})
    judge = make_step(0, "j", "judge", QWEN, definition_column="definitie-v1")
    with pytest.raises(ValueError, match="no definer"):
        scores_from_run(df, [judge], "run")

    definer = make_step(0, "d", "definer", GEMMA)
    with pytest.raises(ValueError, match="lacks the columns"):
        scores_from_run(df, [definer, judge], "run")
    with pytest.raises(ValueError, match="more than one definer"):
        scores_from_run(df, [definer, definer], "run")


def test_scores_of_other_definition_column() -> None:
    steps = [
        make_step(0, "d", "definer", GEMMA),
        make_step(1, "j", "judge", QWEN, definition_column="definitie-v2"),
    ]
    df = pd.DataFrame({"betekenis_pid": [1], **_score_columns("j", [3])})

    scores = scores_from_run(df, steps, "run")

    assert set(scores["system"]) == {"gemma-3-27b-it-FP8-dynamic/definitie-v2"}
    assert scores_from_run(df, steps[:1], "run").empty
