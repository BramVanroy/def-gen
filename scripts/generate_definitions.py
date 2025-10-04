import torch
from pathlib import Path
from datasets import load_dataset
from llm_annotator import Annotator


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

# MODELS_NAMES = {
#     "definer": "casperhansen/llama-3.3-70b-instruct-awq",
#     "feedback": ["casperhansen/llama-3.3-70b-instruct-awq"],
#     "revisor": "casperhansen/llama-3.3-70b-instruct-awq",
#     "judge": [
#         "casperhansen/llama-3.3-70b-instruct-awq",
#         "Qwen/Qwen2.5-72B-Instruct-AWQ",
#     ],
# }

STEPS = [
    {"name": "definer", "task": "definer", "model": "Qwen/Qwen3-0.6B"},
    {"name": "feedback-qwen", "task": "feedback", "model": "Qwen/Qwen3-0.6B"},
    {"name": "feedback-gemma", "task": "feedback", "model": "google/gemma-3-270m-it"},
    {"name": "judge-revisor", "task": "revisor", "model": "Qwen/Qwen3-0.6B"},
    {"name": "judge-qwen", "task": "judge", "model": "Qwen/Qwen3-0.6B"},
    {"name": "judge-gemma", "task": "judge", "model": "google/gemma-3-270m-it"},
]
MODELS_NAMES = {
    "definer": "Qwen/Qwen3-0.6B",
    "feedback": [
        "Qwen/Qwen3-0.6B",
        "google/gemma-3-270m-it",
    ],
    "revisor": "Qwen/Qwen3-0.6B",
    "judge": [
        "Qwen/Qwen3-0.6B",
        "google/gemma-3-270m-it",
    ],
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
        feedback += f"Feedback from reviewer #{reviewer_idx}:\n"
        for col in cols:
            if col in sample:
                feedback += f"- {col.replace(f'_{model}', '')}: {sample[col]}\n"
        feedback += "\n"
    return {"feedback": feedback.strip()}

def prepare_examples(voorbeelden: list[str], word: str, num_examples: int = 2, context_words: int = 50) -> str:
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

def main():
    ds = load_dataset("json", data_files="/vol1/bram/def-gen/data/anw_lemma_definitions.jsonl")["train"]
    ds = ds.filter(
        lambda x: (
            x["woordsoort"] == "substantief"
            and len(x["definitie"].split()) > 5
            and x["lemmavorm"][0].isalpha()
            and x["lemmavorm"][0].islower()
        ),
        num_proc=4
    ).map(
        lambda voorbeelden, word: {"voorbeelden": [vb for vb in voorbeelden if word.lower() in vb.lower()]},
        input_columns=["voorbeelden", "lemmavorm"],
        num_proc=4
    ).filter(
        lambda voorbeelden: len(voorbeelden) >= 2,
        input_columns=["voorbeelden"],
        num_proc=4
    ).remove_columns(["mini_definitie", "woordsoort"]).map(
        prepare_examples,
        input_columns=["voorbeelden", "lemmavorm"],
        num_proc=4
    ).shuffle(seed=42).select(range(8))
    
    feedback_cols = {}
    for task in TEMPLATES.keys():
        models = MODELS_NAMES[task] if isinstance(MODELS_NAMES[task], list) else [MODELS_NAMES[task]]
        prev_model = None
        for model in models:
            short_model_name = model.split("/")[-1]
            pdout = Path(f"/vol1/bram/def-gen/outputs/{task}-{short_model_name}")
            print(f"Processing task '{task}' with model '{model}' - output to {pdout}")
            
            if model != prev_model:
                # Reinitialize annotator if model changes
                anno = Annotator(
                    model_id=model,
                    num_proc=4,
                    max_model_len=2400,
                    max_num_seqs=8,
                    tensor_parallel_size=torch.cuda.device_count(),
                    verbose=True,
                )

            ds = anno.annotate_dataset(
                output_dir=pdout,
                dataset=ds,
                prompt_template_file=TEMPLATES[task]["prompt_file"],
                output_schema_file=TEMPLATES[task]["schema_file"],
                keep_columns=True,
                sampling_params={"max_tokens": 576}
            )
            ds = (
                ds.filter(lambda is_valid, finish_reason: is_valid and finish_reason == "stop", input_columns=["valid_fields", "finish_reason"], num_proc=4)
                .remove_columns(["valid_fields", "finish_reason", "num_tokens", "response"])
                .rename_column("prompted", f"prompted_{task}_{short_model_name}")
            )

            if len(ds) == 0:
                raise ValueError(f"All samples were filtered out after {task} with model {model}!")

            print(f"Dataset now has {len(ds)} samples remaining after task {task} with model {model}.")

            if task == "feedback":
                col_map = {k: f"{k}_{short_model_name}" for k in FEEDBACK_FIELDS if k in ds.column_names}
                feedback_cols[short_model_name] = list(col_map.values())
            else:
                o_fields = OUTPUT_FIELDS[task]
                col_map = {k: v for k, v in o_fields.items() if k in ds.column_names}
            ds = ds.rename_columns(col_map)
            prev_model = model

            print(f"Completed {task} with model {model}.")
            print(f"Dataset now has columns: {ds.column_names}")
        
        if task == "feedback" and feedback_cols:
            ds = ds.map(
                aggregate_feedback,
                num_proc=4,
                fn_kwargs={"feedback_cols": feedback_cols},
            )


if __name__ == "__main__":
    main()
