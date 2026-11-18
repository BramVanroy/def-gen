"""Win rates of writers against ANW from the two-way ranker judgments.

Every item is ranked in two orders (rotation 0 and 1). `outcome_per_item`
combines them with a strict rule: an item counts as a win or a loss only if
all orders agree, and as a tie otherwise. A judge that always prefers the
first position therefore produces ties, not wins. The rates of each single
order are reported as well.
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from def_gen.analysis.stats import bootstrap_ci, ratio_of_sums


OUTCOMES = ["a", "tie", "b"]
PAIR_GROUP = ["judge", "system_a", "system_b"]


def outcome_per_item(pairs: pd.DataFrame) -> pd.DataFrame:
    """Combine the orders of each item and judge into one outcome.

    Args:
        pairs: The pairs table of `AnalysisData`.

    Returns:
        One row per judge, system pair and item with the columns outcome,
        n_orders (the number of orders that were judged) and consistent
        (whether all orders gave the same outcome).

    Examples:
        >>> pairs = pd.DataFrame(
        ...     {
        ...         "judge": "j",
        ...         "system_a": "w",
        ...         "system_b": "anw",
        ...         "item": [1, 1, 2, 2],
        ...         "outcome": ["a", "a", "a", "tie"],
        ...     }
        ... )
        >>> outcome_per_item(pairs)["outcome"].tolist()
        ['a', 'tie']
    """
    grouped = pairs.groupby([*PAIR_GROUP, "item"], sort=True)["outcome"]
    out = grouped.agg(
        n_orders="size", n_distinct="nunique", first="first"
    ).reset_index()
    out["consistent"] = out["n_distinct"] == 1
    out["outcome"] = out["first"].where(out["consistent"], "tie")
    return out.drop(columns=["n_distinct", "first"])


def _sign_test(wins: int, losses: int) -> float:
    if wins + losses == 0:
        return float("nan")
    return float(binomtest(wins, wins + losses, 0.5).pvalue)


def win_rates(
    pairs: pd.DataFrame, num_bootstraps: int = 2000, seed: int | None = 42
) -> pd.DataFrame:
    """Compute win, tie and loss rates of system_a against system_b.

    Rates come with item-bootstrap percentile intervals. The two-sided sign
    test compares wins with losses and leaves ties out. The column `orders`
    is "both" for the combined outcome of `outcome_per_item` and
    "rotation <r>" for the outcome of a single order.

    Args:
        pairs: The pairs table of `AnalysisData`.
        num_bootstraps: Number of bootstrap resamples.
        seed: Seed of the bootstrap.

    Returns:
        One row per judge, system pair and orders with n_items, n_wins,
        n_ties, n_losses, win_rate, tie_rate, loss_rate, their
        `_ci_low`/`_ci_high` bounds and sign_test_p.
    """
    combined = outcome_per_item(pairs).assign(orders="both")
    single = pairs.assign(orders="rotation " + pairs["rotation"].astype(str))
    table = pd.concat([combined, single], ignore_index=True)

    rows = []
    for keys, group in table.groupby([*PAIR_GROUP, "orders"]):
        onehot = np.stack(
            [(group["outcome"] == o).to_numpy() for o in OUTCOMES], axis=1
        ).astype(float)
        rates = onehot.mean(axis=0)
        ci = bootstrap_ci(
            onehot, lambda s: s.mean(axis=1), num_bootstraps, seed
        )
        counts = onehot.sum(axis=0).astype(int)
        row = dict(zip([*PAIR_GROUP, "orders"], keys, strict=True))
        row |= {
            "n_items": len(group),
            "n_wins": counts[0],
            "n_ties": counts[1],
            "n_losses": counts[2],
        }
        for i, name in enumerate(["win", "tie", "loss"]):
            row[f"{name}_rate"] = rates[i]
            row[f"{name}_ci_low"] = ci[0, i]
            row[f"{name}_ci_high"] = ci[1, i]
        row["sign_test_p"] = _sign_test(counts[0], counts[2])
        rows.append(row)
    return pd.DataFrame(rows)


def position_bias(
    pairs: pd.DataFrame, num_bootstraps: int = 2000, seed: int | None = 42
) -> pd.DataFrame:
    """Estimate how often the definition at position 1 wins, per judge.

    Only non-tied pairs count. All system pairs and orders of a judge are
    pooled, and the bootstrap resamples items with all their pairs. The
    binomial test against 0.5 treats the pairs as independent and is
    therefore optimistic. The bootstrap interval accounts for the pairing.

    Args:
        pairs: The pairs table of `AnalysisData`.
        num_bootstraps: Number of bootstrap resamples.
        seed: Seed of the bootstrap.

    Returns:
        One row per judge with n_pairs, n_ties, n_position1_wins,
        n_position2_wins, p_position1 with its `_ci_low`/`_ci_high` bounds
        and binom_test_p.
    """
    rows = []
    for judge, group in pairs.groupby("judge"):
        counts = (
            group.assign(
                pos1=group["winner_position"] == 1,
                decided=group["winner_position"].notna(),
            )
            .groupby("item")[["pos1", "decided"]]
            .sum()
            .to_numpy()
        )
        n_pos1, n_decided = (int(n) for n in counts.sum(axis=0))
        ci = bootstrap_ci(counts, ratio_of_sums, num_bootstraps, seed)
        rows.append(
            {
                "judge": judge,
                "n_pairs": len(group),
                "n_ties": len(group) - n_decided,
                "n_position1_wins": n_pos1,
                "n_position2_wins": n_decided - n_pos1,
                "p_position1": n_pos1 / n_decided
                if n_decided
                else float("nan"),
                "p_position1_ci_low": ci[0],
                "p_position1_ci_high": ci[1],
                "binom_test_p": _sign_test(n_pos1, n_decided - n_pos1),
            }
        )
    return pd.DataFrame(rows)


def order_consistency(
    pairs: pd.DataFrame, num_bootstraps: int = 2000, seed: int | None = 42
) -> pd.DataFrame:
    """Compute the share of comparisons with the same outcome in all orders.

    Outcomes are compared after mapping positions back to systems, so a
    judge that always picks position 1 has a consistency of 0. Comparisons
    that were judged in only one order are counted and left out.

    Args:
        pairs: The pairs table of `AnalysisData`.
        num_bootstraps: Number of bootstrap resamples.
        seed: Seed of the bootstrap.

    Returns:
        One row per judge with n_comparisons (judged in two or more orders),
        n_single_order, n_consistent, n_flips (one order picks a and another
        b), and consistency with its `_ci_low`/`_ci_high` bounds.
    """
    combined = outcome_per_item(pairs)
    order_sets = pairs.groupby([*PAIR_GROUP, "item"], sort=True)[
        "outcome"
    ].agg(set)
    combined["flip"] = [{"a", "b"} <= s for s in order_sets.to_numpy()]
    rows = []
    for judge, group in combined.groupby("judge"):
        multi = group[group["n_orders"] > 1]
        counts = (
            multi.assign(n=1)
            .groupby("item")[["consistent", "n"]]
            .sum()
            .to_numpy()
        )
        ci = bootstrap_ci(counts, ratio_of_sums, num_bootstraps, seed)
        rows.append(
            {
                "judge": judge,
                "n_comparisons": len(multi),
                "n_single_order": len(group) - len(multi),
                "n_consistent": int(multi["consistent"].sum()),
                "n_flips": int(multi["flip"].sum()),
                "consistency": multi["consistent"].mean()
                if len(multi)
                else float("nan"),
                "consistency_ci_low": ci[0],
                "consistency_ci_high": ci[1],
            }
        )
    return pd.DataFrame(rows)
