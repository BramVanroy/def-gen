import pandas as pd
import pytest

from def_gen.analysis.human import human_llm_agreement


def _human(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["item", "rater", "system_a", "system_b", "outcome"]
    )


def _llm(judge: str, rows: list[tuple]) -> pd.DataFrame:
    """Rows are (item, rotation, outcome) for the pair w vs anw."""
    return pd.DataFrame(
        [
            {
                "judge": judge,
                "item": item,
                "rotation": rotation,
                "system_a": "w",
                "system_b": "anw",
                "outcome": outcome,
            }
            for item, rotation, outcome in rows
        ]
    )


def _both(judge: str, outcomes: list[str]) -> pd.DataFrame:
    rows = [(i, r, o) for i, o in enumerate(outcomes) for r in (0, 1)]
    return _llm(judge, rows)


def _row(table: pd.DataFrame, r1: str, r2: str) -> pd.Series:
    sel = table[(table["rater_1"] == r1) & (table["rater_2"] == r2)]
    assert len(sel) == 1
    return sel.iloc[0]


def test_canonicalization_flips_outcomes():
    human = _human(
        [
            (0, "h1", "anw", "w", "a"),
            (1, "h1", "anw", "w", "tie"),
            (0, "h2", "w", "anw", "b"),
            (1, "h2", "w", "anw", "tie"),
        ]
    )
    table = human_llm_agreement(human, _both("j", ["b", "tie"]))
    assert _row(table, "h1", "h2")["percent_agreement"] == 1.0
    assert _row(table, "j", "h1")["percent_agreement"] == 1.0


def test_percent_agreement_and_kinds():
    outcomes = ["a", "b", "tie", "a"]
    human = _human(
        [(i, "h1", "w", "anw", o) for i, o in enumerate(outcomes)]
        + [
            (i, "h2", "w", "anw", o)
            for i, o in enumerate(["a", "b", "tie", "b"])
        ]
    )
    llm = pd.concat(
        [
            _both("j1", ["a", "b", "tie", "a"]),
            _both("j2", ["a", "b", "b", "b"]),
        ]
    )
    table = human_llm_agreement(human, llm)

    assert _row(table, "h1", "h2")["kind"] == "human-human"
    assert _row(table, "h1", "h2")["percent_agreement"] == 0.75
    assert _row(table, "h1", "h2")["n_units"] == 4
    assert _row(table, "j1", "h1")["kind"] == "llm-human"
    assert _row(table, "j1", "h1")["percent_agreement"] == 1.0
    assert _row(table, "j1", "j2")["kind"] == "llm-llm"
    assert _row(table, "j1", "j2")["percent_agreement"] == 0.5
    assert not (
        (table["rater_1"] == "h1") & (table["kind"] == "llm-human")
    ).any()
    assert set(table.columns) >= {"krippendorff_alpha_ordinal", "gwet_ac1"}


def test_llm_orders_are_combined():
    llm = _llm("j", [(0, 0, "a"), (0, 1, "b"), (1, 0, "a"), (1, 1, "a")])
    human = _human([(0, "h", "w", "anw", "tie"), (1, "h", "w", "anw", "a")])
    row = _row(human_llm_agreement(human, llm), "j", "h")
    assert row["percent_agreement"] == 1.0


def test_pairs_without_common_units_are_left_out():
    human = _human([(0, "h1", "w", "anw", "a"), (1, "h2", "w", "anw", "a")])
    table = human_llm_agreement(human, _both("j", ["a", "a"]))
    assert table[table["rater_1"] != "all humans"]["rater_1"].tolist() == [
        "j",
        "j",
    ]
    assert not ((table["rater_1"] == "h1") & (table["rater_2"] == "h2")).any()


def test_all_humans_row():
    human = _human(
        [
            (0, "h1", "w", "anw", "a"),
            (0, "h2", "w", "anw", "a"),
            (0, "h3", "w", "anw", "b"),
            (1, "h1", "w", "anw", "b"),
            (1, "h2", "w", "anw", "b"),
            (2, "h1", "w", "anw", "tie"),
        ]
    )
    table = human_llm_agreement(human, _both("j", ["a", "b", "tie"]))
    row = _row(table, "all humans", "")
    assert row["kind"] == "human-human"
    assert row["n_units"] == 2
    assert pd.isna(row["percent_agreement"])
    assert pd.isna(row["gwet_ac1"])
    assert 0 < row["krippendorff_alpha_ordinal"] < 1


def test_unknown_label_raises():
    human = _human([(0, "h", "w", "anw", "x")])
    with pytest.raises(ValueError, match="Unknown outcome"):
        human_llm_agreement(human, _both("j", ["a"]))


def test_duplicate_rows_raise():
    human = _human([(0, "h", "w", "anw", "a"), (0, "h", "anw", "w", "b")])
    with pytest.raises(ValueError, match="Duplicate"):
        human_llm_agreement(human, _both("j", ["a"]))
