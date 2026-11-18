"""Split the ANW dataset into a dev and an eval set and save it to disk."""

import argparse

from datasets import DatasetDict, load_dataset


def main(
    hub_id: str,
    output_dir: str,
    dev_size: int,
    seed: int,
) -> None:
    """Make a seeded random dev/eval split with one row per meaning.

    Args:
        hub_id: Hub dataset id whose train split is split.
        output_dir: Directory that the DatasetDict is saved to.
        dev_size: Number of rows in the dev split. The rest is eval.
        seed: Seed of the shuffle.
    """
    ds = load_dataset(hub_id)["train"]
    pids = ds["betekenis_pid"]
    assert len(pids) == len(set(pids)), "betekenis_pid is not unique"

    split = ds.train_test_split(test_size=dev_size, seed=seed, shuffle=True)
    splits = DatasetDict({"dev": split["test"], "eval": split["train"]})
    splits.save_to_disk(output_dir)
    print({name: len(part) for name, part in splits.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hub_id",
        default="BramVanroy/anw-15-4-2026-monosemous-nouns-5examples",
    )
    parser.add_argument("--output_dir", default="data/splits/anw-5ex")
    parser.add_argument("--dev_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    main(**vars(parser.parse_args()))
