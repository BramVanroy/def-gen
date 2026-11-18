"""Report input, output and total token counts per annotation step of a dataset.

Rows with an invalid step output are dropped first. The prompts are tokenized
with the given tokenizer.

Example:
    uv run python scripts/check_lengths.py BramVanroy/anw-defgen-9964 \
        --tokenizer RedHatAI/gemma-3-27b-it-FP8-dynamic
"""

import argparse

from datasets import disable_progress_bars, load_dataset
from transformers import AutoTokenizer


disable_progress_bars()


def main(
    hub_id: str,
    tokenizer: str,
    num_proc: int,
) -> None:
    """Print token statistics of every `*_prompted` column.

    Args:
        hub_id: Hub dataset id whose train split is analysed.
        tokenizer: Tokenizer id or path for counting prompt tokens.
        num_proc: Number of workers for filtering and tokenizing.
    """
    ds = load_dataset(hub_id)["train"]
    valid_fields_columns = [
        col for col in ds.column_names if col.endswith("_valid_fields")
    ]

    ds = ds.filter(
        lambda sample: all(sample[col] for col in valid_fields_columns),
        num_proc=num_proc,
    )
    print(ds)

    cols_ending_prompted = [
        col for col in ds.column_names if col.endswith("_prompted")
    ]
    prefixes = sorted({col.rsplit("_", 2)[0] for col in cols_ending_prompted})
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    for prefix in prefixes:
        print(f"Analyzing prefix: {prefix}")
        text_column = prefix + "_prompted"
        prompt_num_tokens = ds.map(
            lambda text: {
                "_num_tokens": [
                    len(token_ids)
                    for token_ids in tokenizer(text)["input_ids"]
                ]
            },
            input_columns=[text_column],
            num_proc=num_proc,
            batched=True,
        )["_num_tokens"]
        output_num_tokens = ds[prefix + "_num_tokens"]
        total_num_tokens = [
            prompt + output
            for prompt, output in zip(
                prompt_num_tokens, output_num_tokens, strict=True
            )
        ]

        print("  Statistics (input tokens):")
        max_num_tokens = max(prompt_num_tokens)
        avg_num_tokens = sum(prompt_num_tokens) / len(prompt_num_tokens)
        median_num_tokens = sorted(prompt_num_tokens)[
            len(prompt_num_tokens) // 2
        ]
        print(f"    Max: {max_num_tokens}")
        print(f"    Avg: {avg_num_tokens:.2f}")
        print(f"    Median: {median_num_tokens}")

        print("  Statistics (output tokens):")
        max_num_tokens = max(output_num_tokens)
        avg_num_tokens = sum(output_num_tokens) / len(output_num_tokens)
        median_num_tokens = sorted(output_num_tokens)[
            len(output_num_tokens) // 2
        ]
        print(f"    Max: {max_num_tokens}")
        print(f"    Avg: {avg_num_tokens:.2f}")
        print(f"    Median: {median_num_tokens}")

        print("  Statistics (total tokens):")
        max_num_tokens = max(total_num_tokens)
        avg_num_tokens = sum(total_num_tokens) / len(total_num_tokens)
        median_num_tokens = sorted(total_num_tokens)[
            len(total_num_tokens) // 2
        ]
        print(f"    Max: {max_num_tokens}")
        print(f"    Avg: {avg_num_tokens:.2f}")
        print(f"    Median: {median_num_tokens}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "hub_id",
        nargs="?",
        default="BramVanroy/anw-defgen-9964",
        help="Hub dataset id.",
    )
    parser.add_argument(
        "--tokenizer",
        default="RedHatAI/gemma-3-27b-it-FP8-dynamic",
        help="Tokenizer id or path.",
    )
    parser.add_argument("--num_proc", type=int, default=8)
    main(**vars(parser.parse_args()))
