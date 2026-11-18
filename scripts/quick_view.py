"""Page through the rows of a finished run in the terminal.

Prints the lemma, examples and definitions of a shuffled row. Press Enter for
the next row and "q" to quit.

Example:
    uv run python scripts/quick_view.py \
        outputs/study/writer-gemma4-26b-a4b/full_process_done
"""

import argparse

from datasets import Dataset


def main(fin: str) -> None:
    """Show the rows of a saved dataset one by one.

    Args:
        fin: Directory of a dataset saved with `save_to_disk`.
    """
    ds = Dataset.load_from_disk(fin).shuffle()
    for i, row in enumerate(ds):
        print(f"Example {i + 1}:")
        print(f"Lemmavorm: {row['lemmavorm']}")
        print(f"Voorbeelden:\n{row['voorbeelden']}")
        print(f"Definitie: {row['originele-definitie']}")
        print(f"Definitie v1: {row['definitie-v1']}")
        if "feedback" in row:
            print(f"Feedback: {row['feedback']}")
            print(f"Samenvatting feedback: {row['samenvatting-feedback']}")
            print(f"Definitie v2: {row['definitie-v2']}")
            print("\n")

        inp = input()
        if inp.lower() in ["q", "quit", "exit"]:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "fin", help="Directory of a dataset saved with `save_to_disk`."
    )
    main(**vars(parser.parse_args()))
