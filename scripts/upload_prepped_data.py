from def_gen.definition_generation import init_data

if __name__ == "__main__":
    ds = init_data(
        num_samples=None,
        min_num_examples=5,
        num_proc=4,
    )

    ds.push_to_hub("BramVanroy/anw-15-4-2026-monosemous-nouns-5examples", private=True)