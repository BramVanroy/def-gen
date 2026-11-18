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
from tests.conftest import make_step, simulate_study, write_run


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
    anw_cov = data.coverage[
        data.coverage["group"] == "anw/Qwen3.6-27B-FP8/betekenis"
    ]
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


def test_head_to_head_run_combines_with_writer_run(tmp_path) -> None:
    gemma, qwen = "definitie-gemma4-26b-a4b", "definitie-qwen36-35b-a3b"
    h2h_steps = [
        make_step(
            0,
            "h2h-q-r0",
            "ranker",
            QWEN,
            post_field_renamer={"ranking": "ranking_h2h_q_r0"},
        ),
        make_step(
            1,
            "h2h-q-r1",
            "ranker",
            QWEN,
            order_rotation=1,
            post_field_renamer={"ranking": "ranking_h2h_q_r1"},
        ),
    ]
    h2h = read_run(
        write_run(
            tmp_path / "head-to-head",
            h2h_steps,
            {
                "betekenis_pid": [1, 2],
                "originele-definitie": ["anw 1", "anw 2"],
                gemma: ["gemma 1", "gemma 2"],
                qwen: ["qwen 1", "qwen 2"],
                "volgorde_h2h-q-r0": [[qwen, gemma]] * 2,
                "volgorde_h2h-q-r1": [[gemma, qwen]] * 2,
                "ranking_h2h_q_r0": ["1 beter dan 2", "evenwaardig"],
                "ranking_h2h_q_r1": ["1 beter dan 2", "2 beter dan 1"],
            },
        )
    )
    writer_steps = [
        make_step(0, "definer-gemma4-26b-a4b", "definer", GEMMA),
        make_step(
            1, "judge-q", "judge", QWEN, definition_column="definitie-v1"
        ),
    ]
    writer = read_run(
        write_run(
            tmp_path / "writer",
            writer_steps,
            {
                "betekenis_pid": [1, 2],
                "originele-definitie": ["anw 1", "anw 2"],
                "definitie-v1": ["gemma 1", "gemma 2"],
                **_score_columns("judge-q", [4, 5]),
            },
        )
    )

    data = combine_runs([writer, h2h])

    rot0 = data.pairs[data.pairs["rotation"] == 0]
    assert set(rot0["system_a"]) == {"gemma4-26b-a4b"}
    assert set(rot0["system_b"]) == {"qwen36-35b-a3b"}
    assert rot0["outcome"].tolist() == ["b", "tie"]
    rot1 = data.pairs[data.pairs["rotation"] == 1]
    assert rot1["outcome"].tolist() == ["a", "b"]
    assert set(data.scores["system"]) == {"gemma4-26b-a4b"}
    assert len(data.definitions) == 6


def test_conflicting_definitions_are_rejected(tmp_path) -> None:
    run = read_run(_anw_run(tmp_path))
    other = read_run(_anw_run(tmp_path / "other"))
    other.definitions["definition"] = "changed"
    other.scores["judge"] = "other-judge"

    with pytest.raises(ValueError, match="definition rows"):
        combine_runs([run, other])


def test_combine_runs_keeps_separate_item_sets_per_table() -> None:
    sim = simulate_study(num_items=10)
    data = AnalysisData(
        scores=sim.scores[sim.scores["item"] < 4],
        pairs=sim.pairs,
        definitions=sim.definitions,
        attrition=sim.attrition,
    )

    combined = combine_runs([data])

    assert set(combined.pairs["item"]) == set(range(10))
    assert set(combined.scores["item"]) == set(range(4))
    dropped = combined.coverage.groupby("table")["n_dropped"].max()
    assert dropped.to_dict() == {"pairs": 0, "scores": 0}
