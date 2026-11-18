"""Run every analysis on combined runs and render a Markdown report."""

import pandas as pd

from def_gen.analysis.bias import length_bias, self_preference
from def_gen.analysis.bt import bradley_terry, writer_mcnemar
from def_gen.analysis.load import ANW_SYSTEM, AnalysisData
from def_gen.analysis.overlap import overlap_table, summarize_overlap
from def_gen.analysis.pairwise import (
    order_consistency,
    position_bias,
    win_rates,
)
from def_gen.analysis.scores import score_agreement, score_differences


METHODS = """\
- Items are the resampling unit of every bootstrap ({num_bootstraps} \
resamples, seed {seed}, 95% percentile intervals).
- Each pair is ranked in two orders. The combined outcome of an item is a \
win or a loss only if both orders agree, and a tie otherwise. The rates of \
each single order are in `win_rates.csv`.
- The sign test compares wins with losses and leaves ties out.
- Bradley-Terry abilities (choix) use the non-tied pairs of both orders as \
separate comparisons and are shifted so that ANW is 0. choix has no tie \
model, so ties are counted separately.
- Score differences are writer minus ANW per item. The Wilcoxon test drops \
zero differences and its p-values are Holm-adjusted per judge.
- Only items present in every comparison are used; see the coverage table.
"""


def run_analysis(
    data: AnalysisData, num_bootstraps: int = 2000, seed: int | None = 42
) -> dict[str, pd.DataFrame]:
    """Compute all result tables from combined runs.

    Analyses whose input is missing (for example scores when only rankers
    ran) give an empty table.

    Args:
        data: Output of `combine_runs`.
        num_bootstraps: Number of bootstrap resamples.
        seed: Seed of the bootstraps.

    Returns:
        A mapping from table name to DataFrame. The names are also the CSV
        file names that `scripts/analyze.py` writes.
    """
    boot = {"num_bootstraps": num_bootstraps, "seed": seed}
    pairs, scores = data.pairs, data.scores
    has_anw_pairs = (pairs["system_b"] == ANW_SYSTEM).any()
    has_anw_scores = (scores["system"] == ANW_SYSTEM).any() and (
        scores["system"] != ANW_SYSTEM
    ).any()
    overlap = overlap_table(data.definitions)
    empty = pd.DataFrame()
    return {
        "coverage": data.coverage,
        "attrition": data.attrition,
        "win_rates": win_rates(pairs, **boot) if len(pairs) else empty,
        "position_bias": position_bias(pairs, **boot) if len(pairs) else empty,
        "order_consistency": order_consistency(pairs, **boot)
        if len(pairs)
        else empty,
        "bradley_terry": bradley_terry(pairs, **boot)
        if has_anw_pairs
        else empty,
        "writer_mcnemar": writer_mcnemar(pairs) if has_anw_pairs else empty,
        "score_differences": score_differences(scores, **boot)
        if has_anw_scores
        else empty,
        "score_agreement": score_agreement(scores, **boot)
        if scores["judge"].nunique() > 1
        else empty,
        "self_preference": self_preference(pairs, scores)
        if has_anw_pairs and has_anw_scores
        else empty,
        "length_bias": length_bias(pairs, data.definitions)
        if has_anw_pairs
        else empty,
        "overlap_items": overlap,
        "overlap_summary": summarize_overlap(overlap),
    }


def _fmt(value: object, digits: int) -> str:
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        if value != 0 and abs(value) < 10**-digits:
            return f"{value:.1e}"
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(df: pd.DataFrame, digits: int = 3) -> str:
    """Render a DataFrame as a Markdown table.

    Args:
        df: The table. Its index is not shown.
        digits: Number of decimals for floats.

    Returns:
        The Markdown table, or "(no data)" for an empty table.

    Examples:
        >>> print(markdown_table(pd.DataFrame({"a": [1], "b": [0.5]}), 2))
        | a | b |
        | --- | --- |
        | 1 | 0.50 |
    """
    if df.empty:
        return "(no data)"
    lines = [
        "| " + " | ".join(map(str, df.columns)) + " |",
        "| " + " | ".join("---" for _ in df.columns) + " |",
    ]
    lines += [
        "| " + " | ".join(_fmt(v, digits) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join(lines)


def _with_ci(df: pd.DataFrame, column: str, prefix: str) -> list[str]:
    """Format "value [low, high]" from a column and its CI columns."""
    return [
        f"{v:.3f} [{lo:.3f}, {hi:.3f}]"
        for v, lo, hi in zip(
            df[column],
            df[f"{prefix}_ci_low"],
            df[f"{prefix}_ci_high"],
            strict=True,
        )
    ]


def _section(title: str, body: str) -> str:
    return f"## {title}\n\n{body}\n"


def render_report(
    tables: dict[str, pd.DataFrame],
    num_bootstraps: int = 2000,
    seed: int | None = 42,
) -> str:
    """Render the main numbers of `run_analysis` as Markdown.

    Args:
        tables: Output of `run_analysis`.
        num_bootstraps: Number of resamples that were used.
        seed: Seed that was used.

    Returns:
        The report text.
    """
    parts = ["# Analysis report\n"]
    parts.append(
        _section(
            "Method",
            METHODS.format(num_bootstraps=num_bootstraps, seed=seed),
        )
    )

    coverage = tables["coverage"]
    attrition = tables["attrition"]
    parts.append(
        _section(
            "Data",
            "Items per comparison before and after the restriction to the"
            " common items:\n\n"
            + markdown_table(coverage)
            + "\n\nInvalid outputs per step:\n\n"
            + markdown_table(
                attrition.drop(
                    columns="invalid_finish_reasons", errors="ignore"
                )
            ),
        )
    )

    rates = tables["win_rates"]
    if not rates.empty:
        both = rates[rates["orders"] == "both"]
        shown = pd.DataFrame(
            {
                "system_a": both["system_a"],
                "system_b": both["system_b"],
                "judge": both["judge"],
                "n": both["n_items"],
                "win": _with_ci(both, "win_rate", "win"),
                "tie": _with_ci(both, "tie_rate", "tie"),
                "loss": _with_ci(both, "loss_rate", "loss"),
                "sign_test_p": both["sign_test_p"],
            }
        )
        single = rates[rates["orders"] != "both"].pivot_table(
            index=["system_a", "system_b", "judge"],
            columns="orders",
            values="win_rate",
        )
        parts.append(
            _section(
                "Win rates (system_a against system_b, both orders combined)",
                markdown_table(shown)
                + "\n\nWin rate of system_a per single order:\n\n"
                + markdown_table(single.reset_index()),
            )
        )

    bias, consistency = tables["position_bias"], tables["order_consistency"]
    if not bias.empty:
        merged = pd.DataFrame(
            {
                "judge": bias["judge"],
                "n_pairs": bias["n_pairs"],
                "p_position1": _with_ci(bias, "p_position1", "p_position1"),
                "binom_test_p": bias["binom_test_p"],
                "consistency": _with_ci(
                    consistency, "consistency", "consistency"
                ),
                "n_flips": consistency["n_flips"],
            }
        )
        parts.append(
            _section(
                "Position bias and order consistency",
                "P(position 1 wins) uses non-tied pairs. Consistency is the"
                " share of comparisons with the same outcome in both"
                " orders.\n\n" + markdown_table(merged),
            )
        )

    bt = tables["bradley_terry"]
    if not bt.empty:
        shown = pd.DataFrame(
            {
                "judge": bt["judge"],
                "system": bt["system"],
                "ability": _with_ci(bt, "ability", "ability"),
                "n_wins": bt["n_wins"],
                "n_losses": bt["n_losses"],
                "n_ties_judge": bt["n_ties"],
            }
        )
        parts.append(
            _section(
                "Bradley-Terry abilities (log-odds, ANW = 0)",
                markdown_table(shown)
                + "\n\nPaired comparison of writers (win against ANW,"
                " exact McNemar):\n\n"
                + markdown_table(tables["writer_mcnemar"]),
            )
        )

    diffs = tables["score_differences"]
    if not diffs.empty:
        shown = pd.DataFrame(
            {
                "judge": diffs["judge"],
                "writer": diffs["writer"],
                "criterion": diffs["criterion"],
                "n": diffs["n_items"],
                "mean_diff": _with_ci(diffs, "mean_diff", "mean_diff"),
                "median_diff": diffs["median_diff"],
                "p_holm": diffs["wilcoxon_p_holm"],
            }
        )
        parts.append(
            _section(
                "Score differences (writer minus ANW)", markdown_table(shown)
            )
        )

    agreement = tables["score_agreement"]
    if not agreement.empty:
        shown = pd.DataFrame(
            {
                "system": agreement["system"],
                "criterion": agreement["criterion"],
                "n_units": agreement["n_units"],
                "alpha": _with_ci(agreement, "alpha", "alpha"),
            }
        )
        parts.append(
            _section(
                "Judge agreement on scores (ordinal Krippendorff's alpha)",
                markdown_table(shown),
            )
        )

    preference = tables["self_preference"]
    if not preference.empty:
        matrix = preference.pivot_table(
            index="writer_family", columns="judge_family", values="win_rate"
        )
        score_matrix = preference.pivot_table(
            index="writer_family",
            columns="judge_family",
            values="mean_score_diff",
        )
        parts.append(
            _section(
                "Self-preference (rows: writer family, columns: judge family)",
                "Win rate against ANW:\n\n"
                + markdown_table(matrix.reset_index())
                + "\n\nMean score difference with ANW:\n\n"
                + markdown_table(score_matrix.reset_index()),
            )
        )

    lengths = tables["length_bias"]
    if not lengths.empty:
        parts.append(
            _section(
                "Length bias (logistic regression of a writer win on the"
                " word-length difference with ANW)",
                markdown_table(lengths),
            )
        )

    parts.append(
        _section(
            "Overlap with the ANW definitions",
            markdown_table(tables["overlap_summary"]),
        )
    )
    return "\n".join(parts)
