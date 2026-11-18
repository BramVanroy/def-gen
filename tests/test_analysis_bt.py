import numpy as np
import pandas as pd
import pytest

from def_gen.analysis.bt import bradley_terry, writer_mcnemar


def _pair_rows(writer: str, outcomes: list[str], judge: str = "j") -> list:
    return [
        {
            "item": item,
            "judge": judge,
            "rotation": 0,
            "system_a": writer,
            "system_b": "anw",
            "position_a": 1,
            "outcome": outcome,
            "winner_position": None,
        }
        for item, outcome in enumerate(outcomes)
    ]


def test_bradley_terry_matches_log_odds_for_one_pair() -> None:
    # With one pair, the MLE ability difference is log(wins / losses).
    pairs = pd.DataFrame(_pair_rows("w", ["a"] * 30 + ["b"] * 10 + ["tie"]))

    bt = bradley_terry(pairs, num_bootstraps=200, alpha=0.0).set_index(
        "system"
    )

    assert bt.loc["anw", "ability"] == 0.0
    assert bt.loc["w", "ability"] == pytest.approx(np.log(3), rel=1e-4)
    assert (
        bt.loc["w", "ability_ci_low"]
        < np.log(3)
        < (bt.loc["w", "ability_ci_high"])
    )
    assert (bt.loc["w", "n_wins"], bt.loc["w", "n_losses"]) == (30, 10)
    assert (bt["n_comparisons"] == 40).all()
    assert (bt["n_ties"] == 1).all()


def test_bradley_terry_orders_simulated_writers(study) -> None:
    bt = bradley_terry(study.pairs, num_bootstraps=100)

    for _, judge in bt.groupby("judge"):
        ability = judge.set_index("system")["ability"]
        assert ability["gemma-3-27b"] > 0 > ability["Qwen3.6-27B"]


def test_bradley_terry_requires_reference() -> None:
    pairs = pd.DataFrame(_pair_rows("w", ["a"]))
    pairs["system_b"] = "v"

    with pytest.raises(ValueError, match="no pair with anw"):
        bradley_terry(pairs, num_bootstraps=10)


def test_writer_mcnemar_matches_hand_computed_value() -> None:
    w1 = ["a"] * 8 + ["b"] + ["a"] * 3 + ["tie"] * 2
    w2 = ["b"] * 8 + ["a"] + ["a"] * 3 + ["b"] * 2
    pairs = pd.DataFrame(_pair_rows("w1", w1) + _pair_rows("w2", w2))

    out = writer_mcnemar(pairs).iloc[0]

    assert (out["writer_1"], out["writer_2"], out["n_items"]) == (
        "w1",
        "w2",
        14,
    )
    assert (out["n_only_1"], out["n_only_2"]) == (8, 1)
    assert out["win_rate_1"] == pytest.approx(11 / 14)
    assert out["mcnemar_p"] == pytest.approx(20 / 512)


def test_writer_mcnemar_uses_common_items_only() -> None:
    rows = _pair_rows("w1", ["a", "a"]) + _pair_rows("w2", ["a"])

    out = writer_mcnemar(pd.DataFrame(rows)).iloc[0]

    assert out["n_items"] == 1
    assert out["mcnemar_p"] == pytest.approx(1.0)
