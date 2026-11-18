"""Export a ranked dataset to Excel for human evaluation in Lexit.

The sheet has `betekenis_pid`, `lemmavorm`, `voorbeelden`, the shuffled
`random-definitie-*` columns and `shuffle-order`, which maps each position
back to its source column.

Example:
    uv run python scripts/prepare_for_lexit.py outputs/run/full_process_done \
        lexit.xlsx --test_split_size 100
"""

from pathlib import Path

from datasets import load_dataset, load_from_disk


def main(
    hub_id_or_local_dir: str,
    output_excel_path: str,
    test_split_size: float | int | None = 100,
    shuffle_seed: int = 42,
):
    """Prepare a dataset for Lexit by selecting relevant columns and exporting to Excel.

    Args:
        hub_id_or_local_dir (str): Hugging Face Hub dataset ID or local directory path.
        output_excel_path (str): Path to save the output Excel file.
        test_split_size (float | int | None): If provided, split the dataset into train/test sets.
            If float, represents the proportion of the dataset to include in the test split.
            If int, represents the absolute number of test samples.
            If None, no split is performed.
        shuffle_seed (int): Seed for shuffling the dataset before splitting/exporting.
    """
    if test_split_size is not None:
        try:
            test_split_size = int(test_split_size)
        except ValueError:
            test_split_size = float(test_split_size)

    if (
        Path(hub_id_or_local_dir).exists()
        and Path(hub_id_or_local_dir).is_dir()
    ):
        ds = load_from_disk(hub_id_or_local_dir)
    else:
        ds = load_dataset(hub_id_or_local_dir)["train"]

    cols = [
        "artikel_pid",
        "betekenis_pid",
        "shuffle-order",
        "lemmavorm",
        "voorbeelden",
        "random-definitie-1",
        "random-definitie-2",
    ]
    if "random-definitie-3" in ds.column_names:
        cols.append("random-definitie-3")

    ds = ds.select_columns(cols).shuffle(shuffle_seed)

    if test_split_size is None:
        ds.to_pandas().to_excel(output_excel_path, index=False)
    else:
        ds_dict = ds.train_test_split(test_size=test_split_size, shuffle=False)
        train_len = len(ds_dict["train"])
        ds_dict["train"].to_pandas().to_excel(
            output_excel_path.replace(".xlsx", f"_{train_len}train.xlsx"),
            index=False,
        )
        test_len = len(ds_dict["test"])
        ds_dict["test"].to_pandas().to_excel(
            output_excel_path.replace(".xlsx", f"_{test_len}test.xlsx"),
            index=False,
        )


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(
        description="Prepare dataset for Lexit by exporting to Excel."
    )
    cparser.add_argument(
        "hub_id_or_local_dir",
        type=str,
        help="Hugging Face Hub dataset ID or local directory path.",
    )
    cparser.add_argument(
        "output_excel_path",
        type=str,
        help="Path to save the output Excel file.",
    )
    cparser.add_argument(
        "--test_split_size",
        type=str,
        default=None,
        help="If provided, split the dataset into train/test sets. If float, represents the proportion of the dataset to include in the test split. "
        "If int, represents the absolute number of test samples. If not provided, no split is performed.",
    )
    cparser.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="Seed for shuffling the dataset before splitting/exporting.",
    )
    cargs = cparser.parse_args()

    main(**vars(cargs))
