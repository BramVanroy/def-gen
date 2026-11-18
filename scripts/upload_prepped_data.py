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
