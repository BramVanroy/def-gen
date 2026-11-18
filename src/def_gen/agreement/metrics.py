"""Agreement metrics between raters."""

from collections.abc import Sequence
from typing import Literal

import krippendorff
import pandas as pd
from statsmodels.stats.inter_rater import (
    aggregate_raters,
    cohens_kappa,
    fleiss_kappa,
)


def calculate_cohen_kappa(df: pd.DataFrame, columns: Sequence[str]) -> float:
    """Calculate Cohen's kappa for two raters over nominal categories.

    Cohen's kappa quantifies agreement between exactly two raters while
    discounting chance overlap. It is suited to nominal (unordered categorical)
    labels, makes no normality assumptions, and should not be used for ordinal
    or interval data without weighting. Rows with a missing value in either
    column are dropped.

    Args:
        df: DataFrame containing the annotations to compare.
        columns: Iterable with the two column names holding each rater's labels.

    Returns:
        float: Chance-corrected agreement score in the range [-1, 1] where:
          * -1 indicates complete disagreement,
          * 0 indicates chance-level agreement, and
          * 1 indicates perfect agreement.

    Raises:
        ValueError: If fewer or more than two columns are supplied.
    """
    if len(columns) != 2:
        raise ValueError("Expected 2 columns to calculate cohen kappa.")

    df = df[list(columns)].dropna()
    categories = df.stack().unique().tolist()
    confusion = pd.crosstab(df[columns[0]], df[columns[1]], dropna=True)
    confusion = confusion.reindex(
        index=categories, columns=categories, fill_value=0
    )
    cohen = cohens_kappa(confusion)
    return float(cohen.kappa)


def calculate_fleiss_kappa(df: pd.DataFrame, columns: Sequence[str]) -> float:
    """Calculate Fleiss' kappa for multiple raters over nominal categories.

    Fleiss' kappa generalizes Cohen's kappa to more than two raters, requiring
    that each item be rated the same number of times, so rows with a missing
    value in any column are dropped. It targets nominal annotations and is
    unsuited for ordered or continuous scales.

    Args:
        df: DataFrame containing rater annotations.
        columns: Iterable of column names providing one column per rater.

    Returns:
        float: Chance-corrected agreement score in the range [-1, 1] where:
          * -1 indicates complete disagreement,
          * 0 indicates chance-level agreement, and
          * 1 indicates perfect agreement.
    """
    df = df[list(columns)].dropna()
    categories = df.stack().unique().tolist()
    cat2id = {c: i for i, c in enumerate(categories)}
    ratings_matrix = df.map(cat2id.get).to_numpy()
    table, _ = aggregate_raters(ratings_matrix)
    fleiss = fleiss_kappa(table, method="fleiss")
    return float(fleiss)


def calculate_krippendorff_alpha(
    df: pd.DataFrame,
    columns: Sequence[str],
    level_of_measurement: Literal[
        "nominal", "ordinal", "interval", "ratio"
    ] = "nominal",
    category_order: Sequence | None = None,
) -> float:
    """Calculate Krippendorff's alpha for any number of raters.

    Missing ratings (NaN or None) are allowed. String labels are mapped to
    integer ids, in the order of `category_order` if given. For "interval" and
    "ratio" the values are used as numbers.

    Args:
        df: DataFrame providing the annotation columns.
        columns: Column names, one per rater.
        level_of_measurement: One of "nominal", "ordinal", "interval" or
            "ratio".
        category_order: Labels from lowest to highest. Required for "ordinal"
            on non-numeric labels, e.g. ["B>A", "A=B", "A>B"]. Ignored for
            "interval" and "ratio".

    Returns:
        float: Chance-corrected agreement score (1 is perfect agreement, 0 is
            chance level). NaN if fewer than two distinct values occur.

    Raises:
        ValueError: If the ordinal level is requested for non-numeric labels
            without `category_order`, or if a label is not in `category_order`.
    """
    data = df[list(columns)]
    if level_of_measurement in ("interval", "ratio"):
        values = data.astype(float)
    elif level_of_measurement == "ordinal" and category_order is None:
        if not all(pd.api.types.is_numeric_dtype(data[c]) for c in data):
            raise ValueError(
                "category_order is required for ordinal data with "
                "non-numeric labels."
            )
        values = data.astype(float)
    else:
        categories = (
            list(category_order)
            if category_order is not None
            else data.stack().dropna().unique().tolist()
        )
        cat2id = {c: i for i, c in enumerate(categories)}
        values = data.map(cat2id.get).astype(float)
        unknown = values.isna() & data.notna()
        if unknown.to_numpy().any():
            bad = data[unknown].stack().unique().tolist()
            raise ValueError(f"Labels not in category_order: {bad}")

    if values.stack().nunique() < 2:
        return float("nan")
    return float(
        krippendorff.alpha(
            reliability_data=values.to_numpy().T,
            level_of_measurement=level_of_measurement,
        )
    )


def calculate_gwet_ac1(df: pd.DataFrame, columns: Sequence[str]) -> float:
    """Calculate Gwet's AC1 for nominal data with two or more raters.

    AC1 is (pa - pe) / (1 - pe). Here pa is the mean over items of the share of
    agreeing rater pairs, and pe = sum_k pi_k * (1 - pi_k) / (q - 1) with pi_k
    the overall share of ratings in category k and q the number of observed
    categories. It is less sensitive than kappa to skewed label distributions.
    Rows with a missing value in any column are dropped.

    Args:
        df: DataFrame containing the annotations to compare.
        columns: Column names, one per rater.

    Returns:
        float: AC1 in [-1, 1], or NaN if no row is complete or only one
            category occurs.

    Examples:
        >>> df = pd.DataFrame(
        ...     {"r1": ["a", "a", "b", "a"], "r2": ["a", "a", "b", "b"]}
        ... )
        >>> round(calculate_gwet_ac1(df, ["r1", "r2"]), 4)
        0.5294
    """
    data = df[list(columns)].dropna().reset_index(drop=True)
    num_raters = len(columns)
    if data.empty or data.stack().nunique() < 2:
        return float("nan")

    counts = pd.crosstab(
        data.index.repeat(num_raters), data.stack().to_numpy()
    )
    pa = (counts * (counts - 1)).sum(axis=1).mean() / (
        num_raters * (num_raters - 1)
    )
    pi = counts.sum(axis=0) / counts.to_numpy().sum()
    pe = float((pi * (1 - pi)).sum() / (len(pi) - 1))
    return float((pa - pe) / (1 - pe))


def calculate_preference_rates(
    df: pd.DataFrame,
    columns: Sequence[str],
    categories: Sequence | None = None,
) -> dict[str, dict]:
    """Calculate how often each rater chose each label.

    Each rater's rates are proportions of that rater's non-missing ratings, so
    they sum to 1 per rater.

    Args:
        df: DataFrame containing the rater annotations.
        columns: Column names, one per rater.
        categories: Labels to report for every rater (with rate 0 if unused).
            Defaults to the labels that occur in any of the columns.

    Returns:
        dict: Mapping from rater column to a mapping from label to rate. A
            rater without any rating gets NaN for every label.

    Examples:
        >>> df = pd.DataFrame({"r1": ["a", "a", "b"], "r2": ["a", None, "a"]})
        >>> calculate_preference_rates(df, ["r1", "r2"])
        {'r1': {'a': 0.6666666666666666, 'b': 0.3333333333333333}, 'r2': {'a': 1.0, 'b': 0.0}}
    """
    if categories is None:
        categories = df[list(columns)].stack().dropna().unique().tolist()
    return {
        col: df[col]
        .value_counts(normalize=True)
        .reindex(categories, fill_value=0.0)
        .to_dict()
        for col in columns
    }


def calculate_percent_agreement(
    df: pd.DataFrame, columns: Sequence[str]
) -> float:
    """Calculate the share of rows in which all raters give the same label.

    Rows with a missing value in any of the columns are excluded, so the
    denominator is the number of complete rows.

    Args:
        df: DataFrame containing the annotations to compare.
        columns: Column names, one per rater.

    Returns:
        float: Proportion in [0, 1], or NaN if no row is complete.

    Examples:
        >>> df = pd.DataFrame({"r1": ["a", "a", "b"], "r2": ["a", "b", None]})
        >>> calculate_percent_agreement(df, ["r1", "r2"])
        0.5
    """
    complete = df[list(columns)].dropna()
    if complete.empty:
        return float("nan")
    return float((complete.nunique(axis=1) == 1).mean())
