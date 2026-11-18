"""Agreement between human raters and LLM judges on pairwise outcomes.

A unit is one comparison of two systems on one item: (item, system_a,
system_b). Human rows are canonicalized like the pairs of `AnalysisData`: the
two systems are sorted with "anw" last, and if a row is swapped, its outcome
"a" becomes "b" and vice versa. Each LLM judge is one rater with the outcome
of `outcome_per_item` (both presentation orders combined).
"""

from itertools import combinations

import pandas as pd

from def_gen.agreement.metrics import (
    calculate_gwet_ac1,
    calculate_krippendorff_alpha,
    calculate_percent_agreement,
)
from def_gen.analysis.pairwise import outcome_per_item


UNIT = ["item", "system_a", "system_b"]
LABELS = {"a", "b", "tie"}
CATEGORY_ORDER = ["b", "tie", "a"]
FLIP = {"a": "b", "b": "a", "tie": "tie"}
ALL_HUMANS = "all humans"
COLUMNS = [
    "rater_1",
    "rater_2",
    "kind",
    "n_units",
    "percent_agreement",
    "krippendorff_alpha_ordinal",
    "gwet_ac1",
]


def _sort_key(system: str) -> tuple[bool, str]:
    return (system == "anw", system)


def _canonicalize_human(human: pd.DataFrame) -> pd.DataFrame:
    unknown = set(human["outcome"]) - LABELS
    if unknown:
        raise ValueError(f"Unknown outcome labels: {sorted(unknown)}")
    swapped = pd.Series(
        [
            _sort_key(a) > _sort_key(b)
            for a, b in zip(human["system_a"], human["system_b"], strict=True)
        ],
        index=human.index,
    )
    out = human.copy()
    out["system_a"] = human["system_a"].where(~swapped, human["system_b"])
    out["system_b"] = human["system_b"].where(~swapped, human["system_a"])
    out["outcome"] = human["outcome"].where(
        ~swapped, human["outcome"].map(FLIP)
    )
    keys = ["rater", *UNIT]
    duplicated = out.duplicated(keys, keep=False)
    if duplicated.any():
        rows = out.loc[duplicated, keys].drop_duplicates()
        raise ValueError(
            f"Duplicate ratings per (rater, item, system pair): "
            f"{rows.to_dict('records')}"
        )
    return out


def _pairwise_metrics(wide: pd.DataFrame, r1: str, r2: str) -> dict | None:
    common = wide[[r1, r2]].dropna()
    if common.empty:
        return None
    columns = [r1, r2]
    return {
        "n_units": len(common),
        "percent_agreement": calculate_percent_agreement(common, columns),
        "krippendorff_alpha_ordinal": calculate_krippendorff_alpha(
            common, columns, "ordinal", CATEGORY_ORDER
        ),
        "gwet_ac1": calculate_gwet_ac1(common, columns),
    }


def human_llm_agreement(
    human: pd.DataFrame, llm_pairs: pd.DataFrame
) -> pd.DataFrame:
    """Compute agreement between human raters and LLM judges.

    Metrics of a rater pair use the units that both raters judged. Ordinal
    Krippendorff alpha uses the order b < tie < a.

    Args:
        human: Long table with the columns item, rater, system_a, system_b
            and outcome ("a", "b" or "tie", relative to the row's system_a).
        llm_pairs: The pairs table of `AnalysisData`.

    Returns:
        One row per rater pair with common units and the columns rater_1,
        rater_2, kind ("human-human", "llm-human" with the LLM as rater_1,
        or "llm-llm"), n_units, percent_agreement,
        krippendorff_alpha_ordinal and gwet_ac1. The row with rater_1
        "all humans" holds the alpha over all humans together and counts the
        units with at least two human ratings.

    Raises:
        ValueError: If an outcome label is unknown or a rater judged the same
            unit twice.

    Examples:
        >>> human = pd.DataFrame(
        ...     {
        ...         "item": [1, 1],
        ...         "rater": ["h1", "h2"],
        ...         "system_a": ["anw", "w"],
        ...         "system_b": ["w", "anw"],
        ...         "outcome": ["b", "a"],
        ...     }
        ... )
        >>> llm = pd.DataFrame(
        ...     columns=["judge", "system_a", "system_b", "item", "outcome"]
        ... )
        >>> human_llm_agreement(human, llm)["percent_agreement"].tolist()[0]
        1.0
    """
    human = _canonicalize_human(human)
    llm = outcome_per_item(llm_pairs)
    human_wide = human.pivot(index=UNIT, columns="rater", values="outcome")
    llm_wide = llm.pivot(index=UNIT, columns="judge", values="outcome")
    wide = human_wide.join(llm_wide, how="outer")

    humans = sorted(human_wide.columns)
    llms = sorted(llm_wide.columns)
    rater_pairs = [
        *((a, b, "llm-llm") for a, b in combinations(llms, 2)),
        *((a, b, "llm-human") for a in llms for b in humans),
        *((a, b, "human-human") for a, b in combinations(humans, 2)),
    ]
    rows = []
    for r1, r2, kind in rater_pairs:
        metrics = _pairwise_metrics(wide, r1, r2)
        if metrics is not None:
            rows.append({"rater_1": r1, "rater_2": r2, "kind": kind} | metrics)

    n_rated = int((human_wide.notna().sum(axis=1) >= 2).sum())
    alpha = float("nan")
    if n_rated:
        alpha = calculate_krippendorff_alpha(
            human_wide, humans, "ordinal", CATEGORY_ORDER
        )
    rows.append(
        {
            "rater_1": ALL_HUMANS,
            "rater_2": "",
            "kind": "human-human",
            "n_units": n_rated,
            "percent_agreement": float("nan"),
            "krippendorff_alpha_ordinal": alpha,
            "gwet_ac1": float("nan"),
        }
    )
    return pd.DataFrame(rows, columns=COLUMNS)
