import numpy as np
import pandas as pd
import pytest

from def_gen.analysis.pairwise import outcome_per_item, win_rates
from def_gen.analysis.stats import bootstrap_ci, ratio_of_sums


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
