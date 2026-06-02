from typing import Iterable

import krippendorff
import pandas as pd
from statsmodels.stats.inter_rater import (
    aggregate_raters,
    cohens_kappa,
    fleiss_kappa,
)


def calculate_cohen_kappa(df: pd.DataFrame, columns: Iterable[str]) -> float:
    """Calculate Cohen's kappa for two raters over nominal categories.

    Cohen's kappa quantifies agreement between exactly two raters while
    discounting chance overlap. It is suited to nominal (unordered categorical)
    labels, makes no normality assumptions, and should not be used for ordinal
    or interval data without weighting.

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

    categories = df[columns].stack().dropna().unique().tolist()
    confusion = pd.crosstab(df[columns[0]], df[columns[1]], dropna=True)
    confusion = confusion.reindex(
        index=categories, columns=categories, fill_value=0
    )
    cohen = cohens_kappa(confusion)
    return float(cohen.kappa)


def calculate_fleiss_kappa(df: pd.DataFrame, columns: Iterable[str]) -> float:
    """Calculate Fleiss' kappa for multiple raters over nominal categories.

    Fleiss' kappa generalizes Cohen's kappa to more than two raters, requiring
    that each item be rated the same number of times. It targets nominal,
    non-normally distributed annotations and is unsuited for ordered or
    continuous scales.

    Args:
        df: DataFrame containing rater annotations.
        columns: Iterable of column names providing one column per rater.

    Returns:
        float: Chance-corrected agreement score in the range [-1, 1] where:
          * -1 indicates complete disagreement,
          * 0 indicates chance-level agreement, and
          * 1 indicates perfect agreement.
    """

    categories = df[columns].stack().dropna().unique().tolist()
    cat2id = {c: i for i, c in enumerate(categories)}
    ratings_matrix = df[columns].map(lambda v: cat2id[v]).to_numpy()
    table, _ = aggregate_raters(ratings_matrix)
    fleiss = fleiss_kappa(table, method="fleiss")
    return fleiss


def calculate_krippendorff_alpha(
    df: pd.DataFrame, columns: Iterable[str]
) -> float:
    """Calculate Krippendorff's alpha for nominal data with flexible rater counts.

    Krippendorff's alpha handles any number of raters, tolerates missing data,
    and supports several measurement levels; here it is computed for nominal
    categories without assuming normality.

    Args:
        df: DataFrame providing the annotation columns.
        columns: Iterable of column names to include in the reliability
            computation.

    Returns:
        float: Chance-corrected agreement score in the range [-1, 1] where:
          * -1 indicates complete disagreement,
          * 0 indicates chance-level agreement, and
          * 1 indicates perfect agreement.
    """
    categories = df[columns].stack().dropna().unique().tolist()
    cat2id = {c: i for i, c in enumerate(categories)}
    ratings_matrix = df[columns].map(lambda v: cat2id[v]).to_numpy()
    alpha = krippendorff.alpha(
        reliability_data=ratings_matrix.T, level_of_measurement="nominal"
    )
    return alpha


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
