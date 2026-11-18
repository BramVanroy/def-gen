from datasets import Dataset


cols = [
    "betekenis",
    "structuur",
    "revisie-nodig",
    "begrijpelijkheid",
    "substitueerbaarheid",
    "gebruik",
    "correctheid",
]


def main(fin: str):
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
    main("/vol1/bram/def-gen/outputs/anw_no_revision/full_process_done")
