import numpy as np
import pandas as pd
import pytest

from def_gen.analysis.pairwise import (
    order_consistency,
    outcome_per_item,
    position_bias,
    win_rates,
)
from def_gen.analysis.stats import bootstrap_ci, ratio_of_sums
from tests.conftest import simulate_study


def _pairs(outcomes_rot0: list[str], outcomes_rot1: list[str]) -> pd.DataFrame:
    rows = []
    for rotation, outcomes in enumerate([outcomes_rot0, outcomes_rot1]):
        for item, outcome in enumerate(outcomes):
            position_a = 2 - rotation
            winner = {"a": position_a, "b": 3 - position_a, "tie": None}
            rows.append(
                {
                    "item": item,
                    "judge": "j",
                    "rotation": rotation,
                    "system_a": "w",
                    "system_b": "anw",
                    "position_a": position_a,
                    "outcome": outcome,
                    "winner_position": winner[outcome],
                }
            )
    return pd.DataFrame(rows)


def test_bootstrap_ci_is_reproducible_and_brackets_the_mean() -> None:
    data = np.arange(20, dtype=float)

    ci = bootstrap_ci(data, lambda s: s.mean(axis=1), 500, seed=1)

    assert ci[0] < data.mean() < ci[1]
    np.testing.assert_array_equal(
        ci, bootstrap_ci(data, lambda s: s.mean(axis=1), 500, seed=1)
    )


def test_bootstrap_ci_handles_empty_and_nan() -> None:
    assert np.isnan(bootstrap_ci(np.array([]), lambda s: s.mean(axis=1))).all()
    assert np.isnan(
        bootstrap_ci(np.ones(3), lambda s: s.mean(axis=1), num_bootstraps=0)
    ).all()
    counts = np.array([[0, 0], [0, 0]])
    assert np.isnan(bootstrap_ci(counts, ratio_of_sums, 50)).all()


def test_outcome_per_item_requires_agreement() -> None:
    pairs = _pairs(["a", "a", "b", "tie"], ["a", "b", "b", "tie"])

    out = outcome_per_item(pairs)

    assert out["outcome"].tolist() == ["a", "tie", "b", "tie"]
    assert out["consistent"].tolist() == [True, False, True, True]
    assert (out["n_orders"] == 2).all()


def test_win_rates_counts_and_sign_test() -> None:
    pairs = _pairs(["a"] * 8 + ["b"] * 2, ["a"] * 8 + ["a"] * 2)

    rates = win_rates(pairs, num_bootstraps=200).set_index("orders")

    both = rates.loc["both"]
    assert (both["n_wins"], both["n_ties"], both["n_losses"]) == (8, 2, 0)
    assert both["win_rate"] == pytest.approx(0.8)
    assert both["win_ci_low"] <= 0.8 <= both["win_ci_high"]
    # Two-sided sign test of 8 wins and 0 losses: 2 * 0.5 ** 8.
    assert both["sign_test_p"] == pytest.approx(2 * 0.5**8)
    assert rates.loc["rotation 0", "n_losses"] == 2
    assert rates.loc["rotation 1", "win_rate"] == 1.0


def test_win_rates_all_ties_has_no_sign_test() -> None:
    rates = win_rates(_pairs(["tie"] * 3, ["tie"] * 3), num_bootstraps=10)

    assert rates["sign_test_p"].isna().all()
    assert (rates["tie_rate"] == 1.0).all()


def test_win_rates_on_simulated_study(study) -> None:
    rates = win_rates(study.pairs, num_bootstraps=200)
    both = rates[rates["orders"] == "both"].set_index(["system_a", "judge"])

    assert len(both) == 6
    assert (both.loc["gemma-3-27b", "win_rate"] > 0.5).all()
    assert (both.loc["Qwen3.6-27B", "win_rate"] < 0.5).all()


def test_position_bias_counts_position_one_wins() -> None:
    # Rotation 0 shows the writer at position 2, rotation 1 at position 1.
    pairs = _pairs(["b", "b", "tie"], ["a", "a", "a"])

    bias = position_bias(pairs, num_bootstraps=100).iloc[0]

    assert (bias["n_pairs"], bias["n_ties"]) == (6, 1)
    assert bias["n_position1_wins"] == 5
    assert bias["p_position1"] == pytest.approx(1.0)
    assert bias["binom_test_p"] == pytest.approx(2 * 0.5**5)


def test_position_bias_without_decided_pairs() -> None:
    bias = position_bias(_pairs(["tie"], ["tie"]), num_bootstraps=10)

    assert np.isnan(bias.loc[0, "p_position1"])
    assert np.isnan(bias.loc[0, "binom_test_p"])


def test_order_consistency() -> None:
    pairs = _pairs(["a", "a", "b", "tie"], ["a", "b", "tie", "tie"])
    single = pd.DataFrame([{**pairs.iloc[0].to_dict(), "item": 9}])

    out = order_consistency(
        pd.concat([pairs, single]), num_bootstraps=100
    ).iloc[0]

    assert (out["n_comparisons"], out["n_single_order"]) == (4, 1)
    assert (out["n_consistent"], out["n_flips"]) == (2, 1)
    assert out["consistency"] == pytest.approx(0.5)
    assert out["consistency_ci_low"] <= 0.5 <= out["consistency_ci_high"]


def test_order_consistency_with_single_orders_only() -> None:
    pairs = _pairs(["a"], ["a"])
    out = order_consistency(pairs[pairs["rotation"] == 0], num_bootstraps=10)

    assert np.isnan(out.loc[0, "consistency"])


def test_simulated_position_bias_is_detected() -> None:
    biased = simulate_study(num_items=80, seed=3, position_bias=3.0)

    bias = position_bias(biased.pairs, num_bootstraps=200)
    consistency = order_consistency(biased.pairs, num_bootstraps=200)

    assert (bias["p_position1_ci_low"] > 0.5).all()
    assert (consistency["consistency"] < 0.5).all()
