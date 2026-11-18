"""Judge bias diagnostics: self-preference and length bias.

Both use the combined outcome of `outcome_per_item` for writer-against-ANW
pairs. Self-preference compares the judge family with the writer family:
a judge that favours its own family has higher values on the diagonal
(same_family is True) than off it.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from def_gen.analysis.load import ANW_SYSTEM, model_family
from def_gen.analysis.pairwise import outcome_per_item


def self_preference(
    pairs: pd.DataFrame, scores: pd.DataFrame, reference: str = ANW_SYSTEM
) -> pd.DataFrame:
    """Tabulate win rates and score differences by writer and judge family.

    Args:
        pairs: The pairs table of `AnalysisData`.
        scores: The scores table of `AnalysisData`.
        reference: The system the writers are compared with.

    Returns:
        One row per writer and judge with writer_family, judge_family,
        same_family, n_items (items with combined pairs), win_rate (combined
        outcome "a", over the pair items) and mean_score_diff (writer minus
        reference, averaged over the scored items and all criteria). The pair
        items and the scored items are separate sets, and the scored items
        can be a subset of the pair items.
    """
    combined = outcome_per_item(pairs[pairs["system_b"] == reference])
    wins = (
        combined.assign(win=combined["outcome"] == "a")
        .groupby(["system_a", "judge"])["win"]
        .agg(n_items="size", win_rate="mean")
        .rename_axis(["writer", "judge"])
    )
    wide = scores.pivot(
        index=["judge", "criterion", "item"], columns="system", values="score"
    )
    diffs = (
        wide.drop(columns=reference)
        .sub(wide[reference], axis=0)
        .groupby(level="judge")
        .mean()
        .stack()
        .rename("mean_score_diff")
        .reorder_levels(["system", "judge"])
        .rename_axis(["writer", "judge"])
    )
    out = pd.concat([wins, diffs], axis=1).reset_index()
    out["writer_family"] = out["writer"].map(model_family)
    out["judge_family"] = out["judge"].map(model_family)
    out["same_family"] = out["writer_family"] == out["judge_family"]
    return out[
        [
            "writer",
            "writer_family",
            "judge",
            "judge_family",
            "same_family",
            "n_items",
            "win_rate",
            "mean_score_diff",
        ]
    ]


def length_bias(
    pairs: pd.DataFrame,
    definitions: pd.DataFrame,
    reference: str = ANW_SYSTEM,
) -> pd.DataFrame:
    """Regress the writer win on the word-length difference, per judge.

    The model is a logistic regression of win (1) against loss (0) on the
    number of words of the writer definition minus that of the reference
    definition, with a fixed effect per writer when there are several.
    Combined ties are left out and counted. Standard errors are clustered by
    item, because one item appears once per writer.

    Args:
        pairs: The pairs table of `AnalysisData`.
        definitions: The definitions table of `AnalysisData`. Only the pair
            items are used.
        reference: The system the writers are compared with.

    Returns:
        One row per judge with n_items, n_ties, coef_length (log-odds per
        word), coef_ci_low, coef_ci_high, p_value and odds_ratio_per_word.
        The estimates are NaN if the fit fails, for example under perfect
        separation.

    Examples:
        A coef_length of 0.1 means that every extra word of the writer
        definition multiplies the odds of a writer win by exp(0.1), about
        1.105.
    """
    words = (
        definitions.set_index(["item", "system"])["definition"]
        .str.split()
        .str.len()
    )
    combined = outcome_per_item(pairs[pairs["system_b"] == reference])
    writer_words = words.reindex(
        pd.MultiIndex.from_arrays([combined["item"], combined["system_a"]])
    )
    reference_words = words.reindex(
        pd.MultiIndex.from_product([combined["item"], [reference]])
    )
    combined["length_diff"] = (
        writer_words.to_numpy() - reference_words.to_numpy()
    )
    if combined["length_diff"].isna().any():
        raise ValueError("Some ranked items lack a definition text.")
    rows = []
    for judge, group in combined.groupby("judge"):
        decided = group[group["outcome"] != "tie"].assign(
            win=lambda d: (d["outcome"] == "a").astype(int)
        )
        formula = "win ~ length_diff"
        if decided["system_a"].nunique() > 1:
            formula += " + C(system_a)"
        row = {
            "judge": judge,
            "n_items": len(decided),
            "n_ties": len(group) - len(decided),
            "coef_length": np.nan,
            "coef_ci_low": np.nan,
            "coef_ci_high": np.nan,
            "p_value": np.nan,
        }
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", PerfectSeparationWarning)
                fit = smf.logit(formula, data=decided).fit(
                    disp=0,
                    cov_type="cluster",
                    cov_kwds={"groups": pd.factorize(decided["item"])[0]},
                )
        except (np.linalg.LinAlgError, PerfectSeparationWarning, ValueError):
            pass
        else:
            ci = fit.conf_int().loc["length_diff"]
            row |= {
                "coef_length": fit.params["length_diff"],
                "coef_ci_low": ci.iloc[0],
                "coef_ci_high": ci.iloc[1],
                "p_value": fit.pvalues["length_diff"],
            }
        row["odds_ratio_per_word"] = np.exp(row["coef_length"])
        rows.append(row)
    return pd.DataFrame(rows)
