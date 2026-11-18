"""Helpers for agreement analysis."""

from collections.abc import Iterable

import pandas as pd


def confusion_matrix(
    df: pd.DataFrame,
    columns: Iterable[str],
    rownames: Iterable[str] | None = None,
    colnames: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compute confusion matrix between two raters.

    Args:
        df (pd.DataFrame): Dataframe containing rater columns.
        columns: The two rater columns to compare.
        rownames: Optional names for the row index.
        colnames: Optional names for the column index.

    Returns:
        pd.DataFrame: Confusion matrix as a DataFrame.
    """
    if len(columns) != 2:
        raise ValueError("Expected 2 columns to compute confusion matrix.")

    conf = pd.crosstab(
        df[columns[0]],
        df[columns[1]],
        rownames=rownames,
        colnames=colnames,
        dropna=False,
    )
    return conf
