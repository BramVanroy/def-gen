import numpy as np
import pandas as pd
import pytest

from def_gen.agreement.bootstrap import bootstrap_agreement
from def_gen.agreement.metrics import (
    calculate_cohen_kappa,
    calculate_fleiss_kappa,
    calculate_gwet_ac1,
    calculate_krippendorff_alpha,
    calculate_percent_agreement,
    calculate_preference_rates,
)
from def_gen.agreement.utils import confusion_matrix


def _two_rater_df() -> pd.DataFrame:
    """Contingency table [[20, 5], [10, 15]] for labels "x" and "y"."""
    pairs = [("x", "x")] * 20 + [("x", "y")] * 5
    pairs += [("y", "x")] * 10 + [("y", "y")] * 15
    return pd.DataFrame(pairs, columns=["r1", "r2"])


def _small_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"r1": ["a", "a", "b", "a"], "r2": ["a", "a", "b", "b"]}
    )


def test_cohen_kappa_matches_hand_computed_value() -> None:
    # po = 0.7, pe = 0.5 -> kappa = 0.4
    assert calculate_cohen_kappa(_two_rater_df(), ["r1", "r2"]) == (
        pytest.approx(0.4)
    )


def test_cohen_kappa_perfect_agreement() -> None:
    df = pd.DataFrame({"r1": ["a", "b", "a", "b"], "r2": ["a", "b", "a", "b"]})

    assert calculate_cohen_kappa(df, ["r1", "r2"]) == pytest.approx(1.0)


def test_cohen_kappa_requires_two_columns() -> None:
    with pytest.raises(ValueError, match="2 columns"):
        calculate_cohen_kappa(_small_df(), ["r1"])


def test_fleiss_kappa_two_raters_matches_hand_computed_value() -> None:
    # po = 0.75, pe = (5/8)^2 + (3/8)^2 = 34/64 -> 7/15
    assert calculate_fleiss_kappa(_small_df(), ["r1", "r2"]) == (
        pytest.approx(7 / 15)
    )


def test_fleiss_kappa_three_raters_perfect_agreement() -> None:
    df = pd.DataFrame(
        {"r1": ["a", "b", "c"], "r2": ["a", "b", "c"], "r3": ["a", "b", "c"]}
    )

    assert calculate_fleiss_kappa(df, ["r1", "r2", "r3"]) == (
        pytest.approx(1.0)
    )


def test_krippendorff_alpha_matches_hand_computed_value() -> None:
    # Do = 0.25, De = 30/56 -> alpha = 8/15
    assert calculate_krippendorff_alpha(_small_df(), ["r1", "r2"]) == (
        pytest.approx(8 / 15)
    )


def test_cohen_kappa_drops_rows_with_missing_values() -> None:
    df = pd.concat(
        [
            _two_rater_df(),
            pd.DataFrame({"r1": [None, "x"], "r2": ["y", None]}),
        ],
        ignore_index=True,
    )

    assert calculate_cohen_kappa(df, ["r1", "r2"]) == pytest.approx(0.4)


def test_fleiss_kappa_drops_rows_with_missing_values() -> None:
    df = pd.concat(
        [_small_df(), pd.DataFrame({"r1": [None, "b"], "r2": ["a", None]})],
        ignore_index=True,
    )

    assert calculate_fleiss_kappa(df, ["r1", "r2"]) == pytest.approx(7 / 15)


def test_krippendorff_alpha_accepts_missing_values() -> None:
    # the last unit has a single rating and is not pairable
    df = pd.DataFrame(
        {"r1": ["a", "a", "b", "a", None], "r2": ["a", "a", "b", "b", "b"]}
    )

    assert calculate_krippendorff_alpha(df, ["r1", "r2"]) == (
        pytest.approx(8 / 15)
    )


def test_krippendorff_alpha_interval_accepts_nan() -> None:
    df = pd.DataFrame({"r1": [1.0, 2.0, np.nan], "r2": [3.0, 2.0, 5.0]})

    assert calculate_krippendorff_alpha(df, ["r1", "r2"], "interval") == (
        pytest.approx(-0.5)
    )


def test_krippendorff_alpha_single_value_is_nan() -> None:
    df = pd.DataFrame({"r1": ["a", "a"], "r2": ["a", "a"]})

    assert np.isnan(calculate_krippendorff_alpha(df, ["r1", "r2"]))


def test_krippendorff_alpha_interval_on_numeric_scores() -> None:
    # units (1, 3) and (2, 2): Do = 2, De = 16/12 -> alpha = 1 - 2 / (4/3)
    df = pd.DataFrame({"r1": [1.0, 2.0], "r2": [3.0, 2.0]})

    assert calculate_krippendorff_alpha(df, ["r1", "r2"], "interval") == (
        pytest.approx(-0.5)
    )


def test_krippendorff_alpha_ordinal_uses_category_order() -> None:
    # ids 0/1/2 with counts n = (2, 3, 3); ordinal distances are
    # d(0,1)^2 = 6.25, d(1,2)^2 = 9, d(0,2)^2 = 30.25
    # Do = 73 / 8, De = 600 / 56 -> alpha = 1 - 511/600 = 89/600
    order = ["B>A", "A=B", "A>B"]
    df = pd.DataFrame(
        {
            "r1": ["B>A", "A=B", "A>B", "A>B"],
            "r2": ["A=B", "A=B", "A>B", "B>A"],
        }
    )

    ordinal = calculate_krippendorff_alpha(
        df, ["r1", "r2"], "ordinal", category_order=order
    )
    swapped = calculate_krippendorff_alpha(
        df, ["r1", "r2"], "ordinal", category_order=["A=B", "B>A", "A>B"]
    )

    assert ordinal == pytest.approx(89 / 600)
    assert swapped != pytest.approx(ordinal)


def test_krippendorff_alpha_ordinal_requires_order_for_strings() -> None:
    with pytest.raises(ValueError, match="category_order"):
        calculate_krippendorff_alpha(_small_df(), ["r1", "r2"], "ordinal")


def test_krippendorff_alpha_rejects_labels_outside_order() -> None:
    with pytest.raises(ValueError, match="not in category_order"):
        calculate_krippendorff_alpha(
            _small_df(), ["r1", "r2"], "ordinal", category_order=["a"]
        )


def test_gwet_ac1_two_raters_matches_hand_computed_value() -> None:
    # pa = 0.75, pi = (5/8, 3/8), pe = 2 * (15/64) / 1 = 15/32
    # AC1 = (3/4 - 15/32) / (17/32) = 9/17
    assert calculate_gwet_ac1(_small_df(), ["r1", "r2"]) == (
        pytest.approx(9 / 17)
    )


def test_gwet_ac1_three_raters_matches_hand_computed_value() -> None:
    # items: aaa, aab -> pa = (1 + 1/3) / 2 = 2/3, pi = (5/6, 1/6)
    # pe = 2 * (5/36) = 5/18 -> AC1 = (2/3 - 5/18) / (13/18) = 7/13
    df = pd.DataFrame({"r1": ["a", "a"], "r2": ["a", "a"], "r3": ["a", "b"]})

    assert calculate_gwet_ac1(df, ["r1", "r2", "r3"]) == pytest.approx(7 / 13)


def test_gwet_ac1_drops_rows_with_missing_values() -> None:
    df = pd.concat(
        [_small_df(), pd.DataFrame({"r1": [None, "b"], "r2": ["a", None]})],
        ignore_index=True,
    )

    assert calculate_gwet_ac1(df, ["r1", "r2"]) == pytest.approx(9 / 17)


def test_gwet_ac1_single_category_is_nan() -> None:
    df = pd.DataFrame({"r1": ["a", "a"], "r2": ["a", "a"]})

    assert np.isnan(calculate_gwet_ac1(df, ["r1", "r2"]))


def test_percent_agreement_matches_hand_computed_value() -> None:
    # 3 of 4 rows agree
    assert calculate_percent_agreement(_small_df(), ["r1", "r2"]) == 0.75


def test_percent_agreement_excludes_rows_with_missing_values() -> None:
    df = pd.DataFrame(
        {"r1": ["a", "a", "b", None], "r2": ["a", "b", None, "b"]}
    )

    # only rows 0 and 1 are complete, one of them agrees
    assert calculate_percent_agreement(df, ["r1", "r2"]) == 0.5


def test_percent_agreement_three_raters_needs_all_to_agree() -> None:
    df = pd.DataFrame({"r1": ["a", "a"], "r2": ["a", "a"], "r3": ["a", "b"]})

    assert calculate_percent_agreement(df, ["r1", "r2", "r3"]) == 0.5


def test_percent_agreement_without_complete_rows_is_nan() -> None:
    df = pd.DataFrame({"r1": ["a", None], "r2": [None, "b"]})

    assert np.isnan(calculate_percent_agreement(df, ["r1", "r2"]))


def test_preference_rates_count_all_raters() -> None:
    rates = calculate_preference_rates(_small_df(), ["r1", "r2"])

    assert rates == {"a": pytest.approx(5 / 8), "b": pytest.approx(3 / 8)}


def test_preference_rates_ignore_missing_values() -> None:
    df = pd.DataFrame({"r1": ["a", None], "r2": ["a", "b"]})

    rates = calculate_preference_rates(df, ["r1", "r2"])

    assert rates == {"a": pytest.approx(2 / 3), "b": pytest.approx(1 / 3)}


def test_confusion_matrix_counts_label_pairs() -> None:
    conf = confusion_matrix(_two_rater_df(), ["r1", "r2"])

    assert conf.loc["x", "x"] == 20
    assert conf.loc["x", "y"] == 5
    assert conf.loc["y", "x"] == 10
    assert conf.loc["y", "y"] == 15


def test_confusion_matrix_custom_names_and_column_check() -> None:
    conf = confusion_matrix(
        _two_rater_df(), ["r1", "r2"], rownames=["A"], colnames=["B"]
    )

    assert conf.index.name == "A"
    assert conf.columns.name == "B"
    with pytest.raises(ValueError, match="2 columns"):
        confusion_matrix(_two_rater_df(), ["r1"])


def _mean_score(df: pd.DataFrame) -> dict[str, float]:
    return {"mean": float(df["score"].mean())}


def test_bootstrap_returns_value_interval_and_metadata() -> None:
    df = pd.DataFrame({"score": np.arange(20, dtype=float)})

    out = bootstrap_agreement(_mean_score, df, num_bootstraps=50, seed=1)

    lo, hi = out["mean"]["ci95"]
    assert out["mean"]["value"] == pytest.approx(9.5)
    assert lo < 9.5 < hi
    assert out["num_bootstraps"] == 50
    assert out["num_items"] == 20
    assert out["bootstrap_seed"] == 1


def test_bootstrap_is_reproducible_for_a_seed() -> None:
    df = pd.DataFrame({"score": np.arange(20, dtype=float)})

    first = bootstrap_agreement(_mean_score, df, num_bootstraps=30, seed=7)
    second = bootstrap_agreement(_mean_score, df, num_bootstraps=30, seed=7)

    assert first == second


def test_bootstrap_ci_keys_restricts_tracked_statistics() -> None:
    df = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0]})

    def fn(d: pd.DataFrame) -> dict[str, float]:
        return {
            "mean": float(d["score"].mean()),
            "max": float(d["score"].max()),
        }

    out = bootstrap_agreement(fn, df, num_bootstraps=5, ci_keys=["max"])

    assert "max" in out
    assert "mean" not in out


def test_bootstrap_rejects_negative_count_and_missing_key() -> None:
    df = pd.DataFrame({"score": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="non-negative"):
        bootstrap_agreement(_mean_score, df, num_bootstraps=-1)
    with pytest.raises(ValueError, match="not found"):
        bootstrap_agreement(
            _mean_score, df, num_bootstraps=2, ci_keys=["missing"]
        )


def test_bootstrap_all_nan_statistic_gives_nan_interval() -> None:
    df = pd.DataFrame({"score": [1.0, 2.0, 3.0]})

    out = bootstrap_agreement(
        lambda d: {"x": float("nan")}, df, num_bootstraps=4
    )

    assert out["x"]["num_skipped"] == 4
    assert np.isnan(out["x"]["ci95"]).all()


def test_bootstrap_counts_skipped_resamples_per_key() -> None:
    df = pd.DataFrame({"score": [0.0, 1.0]})

    def fn(d: pd.DataFrame) -> dict[str, float]:
        # a resample of 2 rows is constant (mean 0 or 1) with probability 1/2
        mean = float(d["score"].mean())
        return {"mean": mean, "variable": mean if mean == 0.5 else np.nan}

    out = bootstrap_agreement(fn, df, num_bootstraps=400, seed=0)

    assert out["mean"]["num_skipped"] == 0
    assert 140 <= out["variable"]["num_skipped"] <= 260
    assert out["variable"]["ci95"] == (0.5, 0.5)
