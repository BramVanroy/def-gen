import numpy as np
import pandas as pd
import pytest
from scipy.stats import wilcoxon

from def_gen.analysis.scores import score_agreement, score_differences


def _scores(values: dict[tuple[str, str], list[float]]) -> pd.DataFrame:
    rows = [
        {
            "item": item,
            "system": system,
            "judge": judge,
            "criterion": "betekenis",
            "score": score,
        }
        for (system, judge), scores in values.items()
        for item, score in enumerate(scores)
        if not np.isnan(score)
    ]
    return pd.DataFrame(rows)


def test_score_differences_hand_example() -> None:
    scores = _scores(
        {
            ("w", "j"): [5, 4, 3, 5, 2, np.nan],
            ("anw", "j"): [4, 4, 1, 3, 3, 5],
        }
    )

    out = score_differences(scores, num_bootstraps=200).iloc[0]

    diff = [1, 0, 2, 2, -1]
    assert (out["n_items"], out["n_unpaired"], out["n_nonzero"]) == (5, 1, 4)
    assert out["mean_diff"] == pytest.approx(0.8)
    assert out["median_diff"] == 1
    assert out["mean_diff_ci_low"] <= 0.8 <= out["mean_diff_ci_high"]
    assert out["wilcoxon_p"] == pytest.approx(wilcoxon(diff).pvalue)
    assert out["wilcoxon_p_holm"] == pytest.approx(out["wilcoxon_p"])
    assert out["mean_reference"] == pytest.approx(3.0)


def test_score_differences_holm_and_zero_differences() -> None:
    scores = _scores(
        {
            ("w", "j"): [5, 5, 5, 5, 5, 5, 5],
            ("v", "j"): [3, 3, 3, 3, 3, 3, 3],
            ("anw", "j"): [3, 3, 3, 3, 3, 3, 3],
        }
    )

    out = score_differences(scores, num_bootstraps=50).set_index("writer")

    assert np.isnan(out.loc["v", "wilcoxon_p"])
    assert np.isnan(out.loc["v", "wilcoxon_p_holm"])
    # A single test in the family: Holm leaves it unchanged.
    assert out.loc["w", "wilcoxon_p_holm"] == pytest.approx(
        out.loc["w", "wilcoxon_p"]
    )


def test_score_differences_on_simulated_study(study) -> None:
    out = score_differences(study.scores, num_bootstraps=100)

    assert len(out) == 3 * 6 * 2
    gemma = out[out["writer"] == "gemma-3-27b"]
    assert (gemma["mean_diff"] > 0).all()
    assert (gemma["wilcoxon_p_holm"] >= gemma["wilcoxon_p"]).all()


def test_score_agreement_perfect_and_pooled() -> None:
    scores = _scores(
        {
            ("w", "j1"): [1, 2, 3, 4, 5],
            ("w", "j2"): [1, 2, 3, 4, 5],
            ("anw", "j1"): [5, 4, 3, 2, 1],
            ("anw", "j2"): [5, 4, 3, 2, np.nan],
        }
    )

    out = score_agreement(scores, num_bootstraps=50).set_index("system")

    assert out.loc["w", "alpha"] == pytest.approx(1.0)
    assert out.loc["all", "n_units"] == 10
    assert out.loc["all", "alpha"] == pytest.approx(1.0)
    assert (out["n_judges"] == 2).all()
