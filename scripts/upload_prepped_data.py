"""Build the filtered ANW dataset with `init_data` and push it to the hub.

Uses `data/anw_lemma_definitions.jsonl`, keeps entries with at least 5 examples
and pushes a private dataset to the author's namespace. Change the dataset id
in the script before running it under another account.

Example:
    uv run python scripts/upload_prepped_data.py
"""

from def_gen.definition_generation import init_data


if __name__ == "__main__":
    MIN_NUM_EXAMPLES = 5
    ds = init_data(
        num_samples=None,
        min_num_examples=MIN_NUM_EXAMPLES,
        num_proc=4,
    )
    print(ds)
    print(ds.features)

    ds.push_to_hub(
        f"BramVanroy/anw-15-4-2026-monosemous-nouns-{MIN_NUM_EXAMPLES}examples",
        private=True,
    )
