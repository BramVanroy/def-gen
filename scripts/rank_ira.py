"""Compute inter-rater agreement between the ranker steps of a dataset.

Example:
    uv run python scripts/rank_ira.py outputs/run/full_process_done \
        --fout agreement.json
"""

import json
from collections.abc import Sequence
from functools import partial

import pandas as pd
from datasets import load_from_disk

from def_gen.agreement.bootstrap import bootstrap_agreement
from def_gen.agreement.metrics import (
    calculate_cohen_kappa,
    calculate_gwet_ac1,
    calculate_krippendorff_alpha,
    calculate_percent_agreement,
    calculate_preference_rates,
)
from def_gen.agreement.utils import confusion_matrix


COL_A = "originele-definitie"
COL_B = "definitie-v1"
CANON_ORDER = [COL_A, COL_B]
ORDINAL_ORDER = ["B>A", "A=B", "A>B"]
COL_LABEL_MAP = {
    COL_A: "A",
    COL_B: "B",
}

LABEL_MAP = {
    "1 beter dan 2": "A>B",
    "2 beter dan 1": "B>A",
    "evenwaardig": "A=B",
}


def agreement_2raters(df: pd.DataFrame, columns: Sequence[str]) -> dict:
    """Compute inter-rater agreement statistics between two raters.

    Args:
        df (pd.DataFrame): Dataframe containing rater columns.
        columns (Sequence[str]): Column names, one per rater.

    Returns:
        dict: Flat mapping from statistic name to value. Preference rates have
            the key "preference_rate/<column>/<label>".
    """
    if len(columns) != 2:
        raise ValueError(f"Expected 2 columns but {len(columns)} were given.")

    stats = {
        "percent_agreement": calculate_percent_agreement(df, columns),
        "cohens_kappa": calculate_cohen_kappa(df, columns),
        "krippendorff_alpha_nominal": calculate_krippendorff_alpha(
            df, columns, "nominal", ORDINAL_ORDER
        ),
        "krippendorff_alpha_ordinal": calculate_krippendorff_alpha(
            df, columns, "ordinal", ORDINAL_ORDER
        ),
        "gwet_ac1": calculate_gwet_ac1(df, columns),
    }
    rates = calculate_preference_rates(df, columns, ORDINAL_ORDER)
    for col, col_rates in rates.items():
        for label, rate in col_rates.items():
            stats[f"preference_rate/{col}/{label}"] = rate
    return stats


def unshuffle_df(
    df: pd.DataFrame,
    shuffle_col: str = "shuffle-order",
    ranking_prefix: str = "ranking_ranker_",
) -> pd.DataFrame:
    """Unshuffle the definitions in the dataframe based on the shuffle order column.

    Args:
        df (pd.DataFrame): Dataframe containing the shuffled rankings and shuffle order.
        shuffle_col (str): Name of the column containing the shuffle order.
        ranking_prefix (str): Prefix of the ranking columns to unshuffle.
    Returns:
        pd.DataFrame: Dataframe with unshuffled rankings.
    """
    df_out = df.copy()

    def unshuffle_row(row):
        order = row[shuffle_col].tolist()
        # The shuffled order is different from what we will use as the default order
        # so switch "2 beter dan 1" and "1 beter dan 2" in the ranker columns
        was_shuffled = None
        for c in row.index:
            if c.startswith(ranking_prefix):
                val = row[c]

                if order == CANON_ORDER:
                    row[f"unshuffled_{c}"] = val
                    if was_shuffled is True:
                        raise ValueError(
                            f"Inconsistent shuffling detected in row: {row}"
                        )
                    was_shuffled = False
                else:
                    if was_shuffled is False:
                        raise ValueError(
                            f"Inconsistent shuffling detected in row: {row}"
                        )
                    if val == "A>B":
                        row[f"unshuffled_{c}"] = LABEL_MAP["2 beter dan 1"]
                    elif val == "B>A":
                        row[f"unshuffled_{c}"] = LABEL_MAP["1 beter dan 2"]
                    elif val == "A=B":
                        row[f"unshuffled_{c}"] = LABEL_MAP["evenwaardig"]
                    else:
                        raise ValueError(
                            f"Unexpected value {val} in column {c} when unshuffling."
                        )
                    was_shuffled = True

        row["was_shuffled"] = was_shuffled
        return row

    df_out = df_out.apply(unshuffle_row, axis=1).drop(columns=[shuffle_col])
    return df_out


def main(
    dir_dataset: str,
    fout: str | None = None,
    num_bootstraps: int = 2000,
    shuffle_order_col: str = "shuffle-order",
    ranking_prefix: str = "ranking_ranker_",
    num_proc: int = 8,
):
    """Given a dataset directory, compute inter-rater agreement statistics and print/save the summary.
    Only suitable for 2 raters.

    Args:
        dir_dataset (str): Path to the dataset directory.
        fout (str | None): Optional path to save the summary as a JSON file.
        num_bootstraps (int): Number of bootstrap samples for confidence intervals.
    """
    ds = load_from_disk(dir_dataset)
    ranking_cols = [c for c in ds.column_names if c.startswith(ranking_prefix)]
    # Rename values in ranking columns to standard labels
    ds = ds.map(
        lambda example: {k: LABEL_MAP[example[k]] for k in ranking_cols},
        num_proc=num_proc,
    )
    ds = ds.select_columns([*ranking_cols, shuffle_order_col])

    df = ds.to_pandas()
    df = unshuffle_df(
        df,
        shuffle_col=shuffle_order_col,
        ranking_prefix=ranking_prefix,
    )

    unshuffled_columns = [f"unshuffled_{c}" for c in ranking_cols]
    agreement_fn = partial(agreement_2raters, columns=unshuffled_columns)
    summary = bootstrap_agreement(
        agreement_fn,
        df,
        num_bootstraps=num_bootstraps,
    )

    print("Agreement summary:")
    print(json.dumps(summary, indent=4))
    print()
    print("Confusion matrix:")
    conf_mat = confusion_matrix(df, columns=unshuffled_columns)
    print(conf_mat)

    if fout:
        with open(fout, "w", encoding="utf-8") as fhout:
            json.dump(summary, fhout, indent=4)
        print(f"\nSaved summary to {fout}")


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(
        description="Compute inter-rater agreement statistics for rankings."
    )
    cparser.add_argument(
        "dir_dataset", type=str, help="Path to the dataset directory."
    )
    cparser.add_argument(
        "--fout",
        type=str,
        default=None,
        help="Optional path to save the summary as a JSON file.",
    )
    cparser.add_argument(
        "--num_bootstraps",
        type=int,
        default=2000,
        help="Number of bootstrap samples for confidence intervals.",
    )
    cargs = cparser.parse_args()

    main(**vars(cargs))
