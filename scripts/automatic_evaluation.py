from datasets import load_from_disk, load_dataset

feedback_suff = ("_llama", "_qwen", "_apertus")
judge_suff = ("_llama", )

def load_ds():    
    orig_ds = load_dataset("json", data_files="/home/local/vanroy/def-gen/data/anw_lemma_definitions.jsonl")["train"]
    orig_ds = orig_ds.filter(
            # Only keep nouns with definitions longer than 5 words and a valid lemmavorm
            # Do not start with a capital letter or non-alphabetic character
            lambda x: (
                x["woordsoort"] == "substantief"
                and len(x["definitie"].split()) > 5
                and x["lemmavorm"][0].isalpha()
                and x["lemmavorm"][0].islower()
            ),
            num_proc=4,
        ).map(
            # Only keep examples that contain the word in the voorbeelden
            lambda voorbeelden, word: {"voorbeelden": [vb for vb in voorbeelden if word.lower() in vb.lower()]},
            input_columns=["voorbeelden", "lemmavorm"],
            num_proc=4,
        ).filter(
            # Only keep examples with at least 2 voorbeelden containing the word
            lambda voorbeelden: len(voorbeelden) >= 2,
            input_columns=["voorbeelden"],
            num_proc=4,
        )
    lemmavorm2definitie = {row["lemmavorm"]: row["definitie"] for row in orig_ds}
    ds = load_from_disk("/home/local/vanroy/def-gen/outputs/6-judge-judge-llama-70b/done")
    ds = ds.map(lambda lemmavorm: {"originele-definitie": lemmavorm2definitie[lemmavorm]}, num_proc=4, input_columns=["lemmavorm"])
    return ds

def main():
    ds = load_ds()
    for example in ds:
        print("=================================================")
        print("WOORD:", example["lemmavorm"])
        print("VOORBEELDEN:")
        print(example["voorbeelden"])
        print("DEFINITIE V1:", example["definitie-v1"])
        print("OORSPRONKELIJK DEFINITIE:", example["originele-definitie"])
        print()

        print("# FEEDBACK")
        for suff in feedback_suff:
            print(f"  ## {suff[1:].upper()}")
            cols = [c for c in example.keys() if c.endswith(suff) and "judge" not in c]
            for col in cols:
                col_str = col.replace(suff, "").replace("_", " ").capitalize()
                print(f"    {col_str}: {example[col]}")
            print()

        print("\n# REVISION")
        print("DEFINITIE V2:", example["definitie-v2"])
        print("OORSPRONKELIJK DEFINITIE:", example["originele-definitie"])

        print("# FINAL JUDGMENT")
        for suff in judge_suff:
            print(f"  ## {suff[1:].upper()}")
            cols = [c for c in example.keys() if c.endswith(suff) and "judge" in c]
            for col in cols:
                col_str = col.replace(suff, "").replace("_", " ").replace(" judge", "").capitalize()
                print(f"    {col_str}: {example[col]}")
            print()
        
        print("=================================================")

if __name__ == "__main__":
    main()