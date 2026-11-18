"""Agreement metrics between raters."""

from collections.abc import Iterable, Sequence
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


def calculate_preference_rates(
    df: pd.DataFrame, columns: Iterable[str]
) -> dict:
    """Calculate preference rates for each category across multiple raters.

    This function computes how often each category was selected by all raters,
    providing insight into overall preferences in the annotations. In other words
    we are aggregating over all raters to see which categories are most frequently chosen.

    Args:
        df: DataFrame containing the rater annotations.
        columns: Iterable of column names representing the raters.

    Returns:
        dict: Mapping from category to its preference rate (proportion of
            total selections).
    """
    flat = df[columns].stack().dropna()
    categories = flat.unique().tolist()
    counts = flat.value_counts().reindex(categories, fill_value=0).astype(int)
    rates = (counts / counts.sum()).to_dict()
    return rates


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
