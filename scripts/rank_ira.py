import json
from collections.abc import Iterable
from functools import partial

import pandas as pd
from datasets import load_from_disk

from def_gen.agreement.bootstrap import bootstrap_agreement
from def_gen.agreement.metrics import (
    calculate_cohen_kappa,
    calculate_fleiss_kappa,
    calculate_krippendorff_alpha,
    calculate_preference_rates,
)
from def_gen.agreement.utils import confusion_matrix


COL_A = "originele-definitie"
COL_B = "definitie-v1"
CANON_ORDER = [COL_A, COL_B]
COL_LABEL_MAP = {
    COL_A: "A",
    COL_B: "B",
}

LABEL_MAP = {
    "1 beter dan 2": "A>B",
    "2 beter dan 1": "B>A",
    "evenwaardig": "A=B",
}


def agreement_2raters(df: pd.DataFrame, columns: Iterable[str]) -> dict:
    """Compute inter-rater agreement statistics for the given dataframe between two raters

    Args:
        df (pd.DataFrame): Dataframe containing rater columns.
        columns (Iterable[str]): Iterable of column names providing one column per rater.
    Returns:
        dict: Dictionary with agreement statistics.
    """
    if len(columns) != 2:
        raise ValueError(f"Expected 2 columns but {len(columns)} were given.")

    cohen_val = calculate_cohen_kappa(df, columns)
    fleiss_val = calculate_fleiss_kappa(df, columns)
    krippendorff_val = calculate_krippendorff_alpha(df, columns)
    preference_val = calculate_preference_rates(df, columns)

    return {
        "cohens_kappa": cohen_val,
        "fleiss_kappa": fleiss_val,
        "krippendorff_alpha_nominal": krippendorff_val,
        "preference_rates": preference_val,
    }


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
    num_bootstraps: int = 1000,
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
        ci_keys=["cohens_kappa", "fleiss_kappa", "krippendorff_alpha_nominal"],
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
        default=1000,
        help="Number of bootstrap samples for confidence intervals.",
    )
    cargs = cparser.parse_args()

    main(**vars(cargs))
