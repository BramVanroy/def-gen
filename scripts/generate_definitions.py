import shutil
from pathlib import Path
from typing import Counter

import torch
from datasets import load_dataset
from llm_annotator import Annotator

from def_gen.pipeline import Pipeline, Step


TEMPLATES = {
    "definer": {
        "prompt_file": Path("/vol1/bram/def-gen/instructions/definer.md"),
        "schema_file": Path("/vol1/bram/def-gen/instructions/definer.schema.json"),
    },
    "feedback":  {
        "prompt_file": Path("/vol1/bram/def-gen/instructions/feedback.md"),
        "schema_file": Path("/vol1/bram/def-gen/instructions/feedback.schema.json"),
    },
    "revisor":  {
        "prompt_file": Path("/vol1/bram/def-gen/instructions/revisor.md"),
        "schema_file": Path("/vol1/bram/def-gen/instructions/revisor.schema.json"),
    },
    "judge":  {
        "prompt_file": Path("/vol1/bram/def-gen/instructions/judge.md"),
        "schema_file": Path("/vol1/bram/def-gen/instructions/judge.schema.json"),
    },
}

FEEDBACK_FIELDS = [
    "begrijpelijkheid",
    "betekenis",
    "gebruik",
    "structuur",
    "substitueerbaarheid",
    "correctheid"
]


def aggregate_feedback(sample: dict, feedback_cols: dict[str, list]):
    feedback = ""
    for reviewer_idx, (model, cols) in enumerate(feedback_cols.items(), 1):
        feedback += f"*Feedback van reviewer #{reviewer_idx}*:\n"
        for col in cols:
            if col in sample:
                feedback += f"- {col.replace(f'_{model}', '')}: {sample[col]}\n"
        feedback += "\n"
    return {"feedback": feedback.strip()}

def prepare_examples(voorbeelden: list[str], word: str, num_examples: int = 2, context_words: int = 25) -> str:
    # For each example, only keep n words before and after the word
    voorbeelden_str = ""
    lc_word = word.lower()
    for vb in voorbeelden[:num_examples]:
        vb = " ".join(vb.strip().split())
        vb_words = vb.split()
        vb_lc_words = vb.lower().split()
        # Use index in lower-case match but extract from original
        # Might be plural or capitalized, so in-check
        word_idx = next((i for i, w in enumerate(vb_lc_words) if lc_word in w), -1)
        start_idx = max(0, word_idx - context_words)
        end_idx = min(len(vb_words), word_idx + context_words + 1)

        vb = "- "
        if start_idx > 0:
            vb += "..."
        vb += " ".join(vb_words[start_idx:end_idx])
        if end_idx < len(vb_words):
            vb += "..."

        voorbeelden_str += vb + "\n"
    return {"voorbeelden": voorbeelden_str.strip()}

def init_data(num_samples: int = 8):
    ds = load_dataset("json", data_files="/vol1/bram/def-gen/data/anw_lemma_definitions.jsonl")["train"]
    ds = ds.filter(
        # Only keep nouns with definitions longer than 5 words and a valid lemmavorm
        # Do not start with a capital letter or non-alphabetic character
        lambda x: (
            x["woordsoort"] == "substantief"
            and len(x["definitie"].split()) > 5
            and x["lemmavorm"][0].isalpha()
            and x["lemmavorm"][0].islower()
        ),
        num_proc=4
    ).map(
        # Only keep examples that contain the word in the voorbeelden
        lambda voorbeelden, word: {"voorbeelden": [vb for vb in voorbeelden if word.lower() in vb.lower()]},
        input_columns=["voorbeelden", "lemmavorm"],
        num_proc=4
    ).filter(
        # Only keep examples with at least 2 voorbeelden containing the word
        lambda voorbeelden: len(voorbeelden) >= 2,
        input_columns=["voorbeelden"],
        num_proc=4
    ).remove_columns(["mini_definitie", "woordsoort"]).map(
        prepare_examples,
        input_columns=["voorbeelden", "lemmavorm"],
        num_proc=4
    ).shuffle(seed=42).select(range(num_samples))

    return ds

def main(num_samples: int = 4, overwrite: bool = False):
    ds = init_data(num_samples=num_samples)
    pipe = Pipeline([
            Step(
                name="definer",
                task="definer",
                model_name="casperhansen/llama-3.3-70b-instruct-awq",
                prompt_file=str(TEMPLATES["definer"]["prompt_file"]),
                schema_file=str(TEMPLATES["definer"]["schema_file"]),
                post_field_renamer={"definitie": "definitie-v1"},
            ),
            Step(
                name="feedback-llama",
                task="feedback",
                model_name="casperhansen/llama-3.3-70b-instruct-awq",
                prompt_file=str(TEMPLATES["feedback"]["prompt_file"]),
                schema_file=str(TEMPLATES["feedback"]["schema_file"]),
                post_field_renamer={k: f"{k}_llama" for k in FEEDBACK_FIELDS},
            ),
            Step(
                name="feedback-gemma",
                task="feedback",
                model_name="pytorch/gemma-3-27b-it-AWQ-INT4",
                prompt_file=str(TEMPLATES["feedback"]["prompt_file"]),
                schema_file=str(TEMPLATES["feedback"]["schema_file"]),
                post_field_renamer={k: f"{k}_gemma" for k in FEEDBACK_FIELDS},
            ),            
            Step(
                name="feedback-qwen",
                task="feedback",
                model_name="Qwen/Qwen3-32B-AWQ",
                prompt_file=str(TEMPLATES["feedback"]["prompt_file"]),
                schema_file=str(TEMPLATES["feedback"]["schema_file"]),
                post_field_renamer={k: f"{k}_qwen" for k in FEEDBACK_FIELDS},
            ),
            Step(
                name="revisor",
                task="revisor",
                model_name="casperhansen/llama-3.3-70b-instruct-awq",
                prompt_file=str(TEMPLATES["revisor"]["prompt_file"]),
                schema_file=str(TEMPLATES["revisor"]["schema_file"]),
                post_field_renamer={"nieuwe-definitie": "definitie-v2"},
            ),
            Step(
                name="judge-llama",
                task="judge",
                model_name="casperhansen/llama-3.3-70b-instruct-awq",
                prompt_file=str(TEMPLATES["judge"]["prompt_file"]),
                schema_file=str(TEMPLATES["judge"]["schema_file"]),
            ),
            Step(
                name="judge-gemma",
                task="judge",
                model_name="pytorch/gemma-3-27b-it-AWQ-INT4",
                prompt_file=str(TEMPLATES["judge"]["prompt_file"]),
                schema_file=str(TEMPLATES["judge"]["schema_file"]),
            ),
            
            Step(
                name="judge-qwen",
                task="judge",
                model_name="Qwen/Qwen3-32B-AWQ",
                prompt_file=str(TEMPLATES["judge"]["prompt_file"]),
                schema_file=str(TEMPLATES["judge"]["schema_file"]),
            ),
        ])

    feedback_cols = {}
    prev_model_name = None
    anno = None
    for step in pipe:
        print(f"Step {step.idx+1}: {step.name} ({step.task}) using model {step.model_name}")
        pdout = Path(f"/vol1/bram/def-gen/outputs/{step.task}-{step.name}")

        if overwrite and pdout.is_dir():
            shutil.rmtree(pdout)

        pdout_done = pdout / "done"

        if pdout_done.is_dir() and any(pdout_done.iterdir()):
            try:
                ds = ds.load_from_disk(pdout_done)
            except Exception:
                shutil.rmtree(pdout_done)
            else:
                print(f"  - Skipping step because output dir {pdout_done} already exists and is non-empty.")
                continue

        pdout_done.mkdir(parents=True, exist_ok=True)

        prefix = f"{step.name}_"

        if step.model_name != prev_model_name:
            # Reinitialize annotator if model changes
            default_init_kwargs = dict(
                model_id=step.model_name,
                num_proc=4,
                max_model_len=2400,
                max_num_seqs=64,
                tensor_parallel_size=torch.cuda.device_count(),
                verbose=True,
            )
            anno = Annotator(
                **{**default_init_kwargs, **step.anno_init_kwargs}
            )
            prev_model_name = step.model_name

        default_run_kwargs = dict(
            output_dir=pdout,
            dataset=ds,
            prompt_template=step.prompt,
            output_schema=step.schema,
            keep_columns=True,
            sampling_params={"max_tokens": 576},
            prefix=prefix,
        )
        ds = anno.annotate_dataset(
            **{**default_run_kwargs, **step.anno_run_kwargs}
        )
        print(ds.column_names, flush=True)
        print("DONE", flush=True)
        # Only keep successful annotations
        num_before = len(ds)
        filtered_ds = (
            ds.filter(lambda is_valid: is_valid, input_columns=[f"{prefix}valid_fields"], num_proc=4)
            .remove_columns([f"{prefix}valid_fields", f"{prefix}finish_reason", f"{prefix}num_tokens", f"{prefix}response"])
        )
        num_after = len(filtered_ds)
        if num_after == 0:
            raise ValueError(f"All samples were filtered out after {step.name} with model {step.model_name}!")
        elif num_after < num_before:
            invalid_ds = ds.filter(lambda is_valid: not is_valid, input_columns=[f"{prefix}valid_fields"], num_proc=4).select_columns(f"{prefix}finish_reason")
            print(
                f"{num_before - num_after} samples were invalid after {step.name} with model {step.model_name}."
                 " Dataset now has {num_after} samples remaining."
            )
            print(
                "A potential cause is that the model could not respond with the full schema due to"
                " 'sampling_params.max_tokens' being too low. The stopping reasons for the invalid"
                " samples are given below (valid='stop')."
            )
            counts = Counter(invalid_ds[f"{prefix}finish_reason"])
            counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
            for reason, count in counts.items():
                print(f"- {reason}: {count:,}")
            del invalid_ds

        # Re-assign to `ds` for next step
        ds = filtered_ds.rename_columns(step.post_field_renamer)

        if step.task == "feedback":
            feedback_cols[step.name] = list(step.post_field_renamer.values())
            if step.last_of_task:
                ds = ds.map(
                    aggregate_feedback,
                    num_proc=4,
                    fn_kwargs={"feedback_cols": feedback_cols},
                )
                feedback_cols = {}

        ds.save_to_disk(pdout_done)

if __name__ == "__main__":
    main()
