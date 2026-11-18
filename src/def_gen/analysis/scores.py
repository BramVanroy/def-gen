"""Per-criterion judge scores: paired differences and judge agreement.

Scores are treated as ordinal (1 to 5). The paired difference of an item is
the writer score minus the ANW score from the same judge on the same
criterion. The Wilcoxon signed-rank test drops items with a zero difference
(scipy's default "wilcox" method), so `n_nonzero` is reported next to it.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from def_gen.agreement.bootstrap import bootstrap_agreement
from def_gen.agreement.metrics import calculate_krippendorff_alpha
from def_gen.analysis.load import ANW_SYSTEM
from def_gen.analysis.stats import bootstrap_ci


def _holm(p_values: pd.Series) -> pd.Series:
    """Holm-adjust the non-missing p-values and keep NaN where missing."""
    adjusted = pd.Series(np.nan, index=p_values.index)
    present = p_values.notna()
    if present.any():
        adjusted[present] = multipletests(p_values[present], method="holm")[1]
    return adjusted


def score_differences(
    scores: pd.DataFrame,
    num_bootstraps: int = 2000,
    seed: int | None = 42,
    reference: str = ANW_SYSTEM,
) -> pd.DataFrame:
    """Compare writer scores with reference scores per judge and criterion.

    Only items with a score for both systems are used; `n_unpaired` counts
    the others. The Holm correction runs over all writers and criteria of
    one judge.

    Args:
        scores: The scores table of `AnalysisData`.
        num_bootstraps: Number of bootstrap resamples.
        seed: Seed of the bootstrap.
        reference: The system that is subtracted.

    Returns:
        One row per judge, criterion and writer with n_items, n_unpaired,
        mean_writer, mean_reference, mean_diff, median_diff (each difference
        with `_ci_low`/`_ci_high` bounds), n_nonzero, wilcoxon_p and
        wilcoxon_p_holm.
    """
    wide = scores.pivot(
        index=["judge", "criterion", "item"], columns="system", values="score"
    )
    writers = [s for s in wide.columns if s != reference]
    rows = []
    for (judge, criterion), table in wide.groupby(
        level=["judge", "criterion"]
    ):
        for writer in writers:
            both = table[[writer, reference]].dropna()
            present = table[[writer, reference]].notna().any(axis=1)
            diff = (both[writer] - both[reference]).to_numpy(dtype=float)
            mean_ci = bootstrap_ci(
                diff, lambda s: s.mean(axis=1), num_bootstraps, seed
            )
            median_ci = bootstrap_ci(
                diff, lambda s: np.median(s, axis=1), num_bootstraps, seed
            )
            n_nonzero = int((diff != 0).sum())
            rows.append(
                {
                    "judge": judge,
                    "criterion": criterion,
                    "writer": writer,
                    "n_items": len(diff),
                    "n_unpaired": int(present.sum()) - len(diff),
                    "mean_writer": both[writer].mean(),
                    "mean_reference": both[reference].mean(),
                    "mean_diff": diff.mean() if len(diff) else np.nan,
                    "mean_diff_ci_low": mean_ci[0],
                    "mean_diff_ci_high": mean_ci[1],
                    "median_diff": np.median(diff) if len(diff) else np.nan,
                    "median_diff_ci_low": median_ci[0],
                    "median_diff_ci_high": median_ci[1],
                    "n_nonzero": n_nonzero,
                    "wilcoxon_p": float(wilcoxon(diff).pvalue)
                    if n_nonzero
                    else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out["wilcoxon_p_holm"] = out.groupby("judge")["wilcoxon_p"].transform(
        _holm
    )
    return out


def score_agreement(
    scores: pd.DataFrame, num_bootstraps: int = 2000, seed: int | None = 42
) -> pd.DataFrame:
    """Compute ordinal Krippendorff's alpha between the judges.

    A unit is one item of one system. Agreement is computed per system and
    criterion, and per criterion over all systems together (system "all").
    Units that a judge did not score are missing values for alpha.

    Args:
        scores: The scores table of `AnalysisData`.
        num_bootstraps: Number of bootstrap resamples (over units).
        seed: Seed of the bootstrap.

    Returns:
        One row per system and criterion with n_units, n_judges, alpha,
        alpha_ci_low and alpha_ci_high.
    """
    wide = scores.pivot(
        index=["criterion", "system", "item"], columns="judge", values="score"
    )
    judges = list(wide.columns)

    def alpha_fn(df: pd.DataFrame) -> dict[str, float]:
        return {"alpha": calculate_krippendorff_alpha(df, judges, "ordinal")}

    groups = [
        (criterion, system, table)
        for (criterion, system), table in wide.groupby(
            level=["criterion", "system"]
        )
    ]
    groups += [
        (criterion, "all", table)
        for criterion, table in wide.groupby(level="criterion")
    ]
    rows = []
    for criterion, system, table in groups:
        result = bootstrap_agreement(
            alpha_fn, table.reset_index(drop=True), num_bootstraps, seed
        )["alpha"]
        rows.append(
            {
                "system": system,
                "criterion": criterion,
                "n_units": len(table),
                "n_judges": len(judges),
                "alpha": result["value"],
                "alpha_ci_low": result["ci95"][0],
                "alpha_ci_high": result["ci95"][1],
            }
        )
    return pd.DataFrame(rows)
