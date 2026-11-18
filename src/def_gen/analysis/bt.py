"""Bradley-Terry abilities of the systems and paired writer comparisons.

The Bradley-Terry model uses the non-tied pairs of every order as separate
comparisons. choix has no tie model (such as Davidson's), so ties are
left out of the fit and reported as counts. Abilities are on the log-odds
scale and shifted so that the reference system ("anw") has ability 0: a
writer with ability 0.5 beats ANW with probability 1 / (1 + exp(-0.5)),
about 0.62.
"""

from itertools import combinations

import choix
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from def_gen.analysis.load import ANW_SYSTEM
from def_gen.analysis.pairwise import outcome_per_item
from def_gen.analysis.stats import bootstrap_ci


def _fit(comparisons: np.ndarray, reference: int, alpha: float) -> np.ndarray:
    """Fit abilities on a win-count matrix and shift the reference to 0."""
    params = choix.ilsr_pairwise_dense(comparisons, alpha=alpha)
    return params - params[reference]


def bradley_terry(
    pairs: pd.DataFrame,
    num_bootstraps: int = 2000,
    seed: int | None = 42,
    alpha: float = 0.01,
    reference: str = ANW_SYSTEM,
) -> pd.DataFrame:
    """Fit a Bradley-Terry model per judge with item-bootstrap intervals.

    A small L2 penalty (`alpha`) keeps the fit finite when a resample has a
    system that never wins or never loses.

    Args:
        pairs: The pairs table of `AnalysisData`.
        num_bootstraps: Number of bootstrap resamples.
        seed: Seed of the bootstrap.
        alpha: Regularization strength of `choix.ilsr_pairwise_dense`.
        reference: System whose ability is fixed at 0.

    Returns:
        One row per judge and system with ability, ability_ci_low,
        ability_ci_high, n_wins and n_losses of the system, and the judge's
        n_comparisons (non-tied pairs) and n_ties.

    Raises:
        ValueError: If a judge has no pair with the reference system.
    """
    rows = []
    for judge, group in pairs.groupby("judge"):
        systems = sorted(set(group["system_a"]) | set(group["system_b"]))
        if reference not in systems:
            raise ValueError(f"Judge {judge!r} has no pair with {reference}.")
        index = {s: i for i, s in enumerate(systems)}
        decided = group[group["outcome"] != "tie"]
        a_wins = decided["outcome"] == "a"
        winners = np.where(a_wins, decided["system_a"], decided["system_b"])
        losers = np.where(a_wins, decided["system_b"], decided["system_a"])

        items = pd.Index(group["item"].unique())
        counts = np.zeros((len(items), len(systems), len(systems)))
        np.add.at(
            counts,
            (
                items.get_indexer(decided["item"]),
                [index[s] for s in winners],
                [index[s] for s in losers],
            ),
            1,
        )
        ref = index[reference]
        abilities = _fit(counts.sum(axis=0), ref, alpha)
        ci = bootstrap_ci(
            counts,
            lambda s, ref=ref: np.stack(
                [_fit(m.sum(axis=0), ref, alpha) for m in s]
            ),
            num_bootstraps,
            seed,
        )
        wins = counts.sum(axis=(0, 2))
        losses = counts.sum(axis=(0, 1))
        rows += [
            {
                "judge": judge,
                "system": system,
                "ability": abilities[i],
                "ability_ci_low": ci[0, i],
                "ability_ci_high": ci[1, i],
                "n_wins": int(wins[i]),
                "n_losses": int(losses[i]),
                "n_comparisons": len(decided),
                "n_ties": len(group) - len(decided),
            }
            for i, system in enumerate(systems)
        ]
    return pd.DataFrame(rows)


def writer_mcnemar(pairs: pd.DataFrame) -> pd.DataFrame:
    """Compare two writers on the items where both were ranked against ANW.

    For each judge and each pair of writers, the indicator is whether the
    combined outcome of `outcome_per_item` is a win against the reference
    (a tie or a loss counts as no win). The exact McNemar test uses the
    items where exactly one writer wins.

    Args:
        pairs: The pairs table of `AnalysisData`. Only pairs with "anw" as
            system_b are used.

    Returns:
        One row per judge and writer pair with n_items, win_rate_1,
        win_rate_2, n_only_1 (items that only writer 1 wins), n_only_2 and
        mcnemar_p.

    Examples:
        If writer 1 wins 8 items that writer 2 loses and writer 2 wins 1 item
        that writer 1 loses, the exact two-sided p-value is that of 1 or fewer
        successes in 9 fair coin flips, doubled: 20 / 512, about 0.039.
    """
    combined = outcome_per_item(pairs[pairs["system_b"] == ANW_SYSTEM])
    combined["win"] = combined["outcome"] == "a"
    wide = combined.pivot_table(
        index=["judge", "item"], columns="system_a", values="win"
    )
    rows = []
    for judge, table in wide.groupby(level="judge"):
        for writer_1, writer_2 in combinations(sorted(table.columns), 2):
            both = table[[writer_1, writer_2]].dropna().astype(bool)
            win_1, win_2 = both[writer_1], both[writer_2]
            only_1 = int((win_1 & ~win_2).sum())
            only_2 = int((~win_1 & win_2).sum())
            result = mcnemar(
                [
                    [int((win_1 & win_2).sum()), only_1],
                    [only_2, int((~win_1 & ~win_2).sum())],
                ],
                exact=True,
            )
            rows.append(
                {
                    "judge": judge,
                    "writer_1": writer_1,
                    "writer_2": writer_2,
                    "n_items": len(both),
                    "win_rate_1": win_1.mean(),
                    "win_rate_2": win_2.mean(),
                    "n_only_1": only_1,
                    "n_only_2": only_2,
                    "mcnemar_p": float(result.pvalue),
                }
            )
    return pd.DataFrame(rows)
