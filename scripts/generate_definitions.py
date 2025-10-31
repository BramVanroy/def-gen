import hashlib
import json
import re
import shutil
from functools import partial
from random import Random
from typing import Counter

import numpy as np

# enable_quiet_mode()
import torch
from datasets import concatenate_datasets, load_dataset
from llm_annotator import Annotator

from def_gen.pipeline import Pipeline, Step
from def_gen.utils import ROOT, PathCompatibleEncoder


JUDGE_FIELDS = ["begrijpelijkheid", "betekenis", "gebruik", "structuur", "substitueerbaarheid", "correctheid"]
FEEDBACK_FIELDS = JUDGE_FIELDS + ["revisie-nodig"]
SAMPLING_PARAMS = {
    # https://huggingface.co/google/gemma-3-12b-it/discussions/25
    "gemma3": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "presence_penalty": 0.1,
    },
    "qwen3": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.1,
    },
    "mistral3": {
        "temperature": 0.15,
        "presence_penalty": 0.1,
    },
}


def set_seed(seed: int = 42):
    # Set random, troch, numpy, cuda, etc. seed
    Random(seed).seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def concat_feedback(sample: dict, feedback_cols: dict[str, list]):
    feedback = ""
    for reviewer_idx, (_, cols) in enumerate(feedback_cols.items(), 1):
        revisie_vereist_name = next((c for c in cols if c.startswith("revisie-nodig")), None)
        if revisie_vereist_name is not None and not sample[revisie_vereist_name]:
            continue

        reviewer_feedback = f"*Feedback van reviewer {reviewer_idx}*\n"
        reviewer_points = []
        for col in cols:
            if col in sample and "revisie-nodig" not in col and sample[col] is not None:
                colname = next((f for f in FEEDBACK_FIELDS if col.startswith(f)), col).lower()
                # Skip if a character is sequentially repeated 3 or more times (1+2)
                # Indicates hallucinations
                if bool(re.search(r"([^.])\1{2,}", sample[col])):
                    print(f"  - Skipping feedback for column {col} due to repeated characters: {sample[col]}")
                    continue
                reviewer_points.append(f"- {colname}: {sample[col]}")

        if reviewer_points:
            feedback += reviewer_feedback + "\n".join(reviewer_points) + "\n\n"

    if feedback == "":
        print(f"Warning: No feedback generated for sample with lemmavorm '{sample['lemmavorm']}'")
        print(f"Feedback columns: {feedback_cols}")
        print("Sample data:")
        for k, v in sample.items():
            if not k.endswith(("_prompted", "_response", "_pid", "_finish_reason", "_num_tokens")):
                print(f"- {k}: {v}")
        print()

    return {"feedback": feedback.strip()}


def prepare_examples(voorbeelden: list[str], word: str, num_examples: int = 2, context_words: int = 45) -> str:
    # For each example, only keep n words before and after the word
    voorbeelden_str = ""
    lc_word = word.lower()
    Random(42).shuffle(voorbeelden)

    for vb in voorbeelden[:num_examples]:
        # Remove extra whitespace/newlines and normalize to spaces
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
            vb += "[...] "
        vb += " ".join(vb_words[start_idx:end_idx])
        if end_idx < len(vb_words):
            vb += " [...]"

        voorbeelden_str += vb + "\n"
    return {"voorbeelden": voorbeelden_str.strip()}


def shuffle_definitions(sample: dict) -> dict:
    if "definitie-v2" in sample:
        # "revision" stage was done
        defs = [sample["definitie-v1"], sample["definitie-v2"], sample["originele-definitie"]]
        names = ["definitie-v1", "definitie-v2", "originele-definitie"]
    else:
        defs = [sample["definitie-v1"], sample["originele-definitie"]]
        names = ["definitie-v1", "originele-definitie"]
    combined = list(zip(defs, names))

    # Deterministic shuffle based on sample ID
    seed = int(hashlib.sha256(str(sample).encode("utf-8")).hexdigest(), 16) % (2**32)
    Random(seed).shuffle(combined)
    defs, names = zip(*combined)
    output = {
        "random-definitie-1": defs[0],
        "random-definitie-2": defs[1],
    }
    if len(defs) == 3:
        output["random-definitie-3"] = defs[2]

    output["shuffle-order"] = list(names)
    return output


def init_data(num_samples: int | None = 8, min_num_examples: int = 2, num_proc: int | None = None):
    if num_samples is not None and min_num_examples < 1:
        raise ValueError("min_num_examples must be at least 1")

    ds = load_dataset("json", data_files=[str(ROOT.joinpath("data/anw_lemma_definitions.jsonl"))])["train"]
    ds = (
        ds.filter(
            # Only keep nouns with definitions longer than 5 words and a valid lemmavorm
            # Do not start with a capital letter or non-alphabetic character
            # Only keep non-polysemous words for now
            lambda x: (
                x["woordsoort"] == "substantief"
                and len(x["definitie"].split()) > 5
                and x["lemmavorm"][0].isalpha()
                and x["lemmavorm"][0].islower()
                and not x["artikel_meerdere_betekenissen"]
            ),
            num_proc=num_proc,
        )
        .map(
            # Only keep examples that contain the word in the voorbeelden (might be pluralized or capitalized)
            lambda voorbeelden, word: {"voorbeelden": [vb for vb in voorbeelden if word.lower() in vb.lower()]},
            input_columns=["voorbeelden", "lemmavorm"],
            num_proc=num_proc,
        )
        .filter(
            # Only keep examples with at least $n$ voorbeelden containing the word
            lambda voorbeelden: len(voorbeelden) >= min_num_examples,
            input_columns=["voorbeelden"],
            num_proc=num_proc,
        )
        .remove_columns(
            ["mini_definitie", "woordsoort", "artikel_id", "artikel_meerdere_betekenissen", "betekenis_id"]
        )
        .rename_column("definitie", "originele-definitie")
        .map(
            prepare_examples,
            input_columns=["voorbeelden", "lemmavorm"],
            num_proc=num_proc,
            fn_kwargs={"num_examples": min_num_examples},
        )
        .shuffle(seed=42)
    )

    if num_samples is not None and num_samples > 0 and num_samples < len(ds):
        ds = ds.select(range(num_samples))

    return ds


def check_rank_validity(sample: dict, plaatsen_cols: list[str]) -> dict:
    if len(plaatsen_cols) != 3:
        raise ValueError("Expected 3 rank positions for validity check (eerste, tweede, derde plaats).")

    rank_values = {"Definitie 1", "Definitie 2", "Definitie 3"}

    # Iterate to check that all rank values (Definitie 1/2/3) are present exactly once
    for col_name in plaatsen_cols:
        col_value = sample[col_name]
        if col_value is None:
            continue
        # The col value has to be present and valid, and each "Definitie X" can only appear once
        if col_value not in rank_values:
            return False
        rank_values.remove(col_value)

    # There should be no rank values left if all were present exactly once
    return len(rank_values) == 0


def main(
    num_samples: int | None = 512,
    min_num_examples: int = 3,
    overwrite: bool = False,
    default_max_num_seqs: int = 128,
    defaul_max_model_len: int = 2400,
    default_max_tokens: int = 256,
    default_num_batched_tokens: int = 12288,
    greedy: bool = False,  # might lead to invalid JSON due to repeated \t\n
    gpu_memory_utilization: float = 0.95,
    num_proc: int = 10,
    root_pdout=ROOT.joinpath("outputs"),
    # Can include {num_samples} which will be replaced with the actual number of samples
    new_hub_id: str = "BramVanroy/anw-defgen-{num_samples}",
    add_revision_stage: bool = False,
):
    set_seed()
    if add_revision_stage:
        RANKER_FIELDS = ["argumentatie", "eerste-plaats", "tweede-plaats", "derde-plaats"]
    else:
        RANKER_FIELDS = ["argumentatie", "ranking"]

    ds = init_data(num_samples=num_samples, min_num_examples=min_num_examples, num_proc=num_proc)
    steps = [
        Step(
            name="definer-gemma-27b",
            task="definer",
            model_name="RedHatAI/gemma-3-27b-it-FP8-dynamic",
            file_directory=ROOT.joinpath("instructions/definer"),
            post_field_renamer={"definitie": "definitie-v1"},
            anno_init_kwargs={"max_model_len": 2048},
            anno_run_kwargs={"sampling_params": {**SAMPLING_PARAMS["gemma3"], "max_tokens": 384}},
        )
    ]
    if add_revision_stage:
        steps.extend(
            [
                Step(
                    name="feedback-gemma-27b",
                    task="feedback",
                    model_name="RedHatAI/gemma-3-27b-it-FP8-dynamic",
                    file_directory=ROOT.joinpath("instructions/feedback"),
                    post_field_renamer={k: f"{k}_gemma" for k in FEEDBACK_FIELDS},
                    anno_init_kwargs={"max_model_len": 3000},
                    anno_run_kwargs={"sampling_params": {**SAMPLING_PARAMS["gemma3"], "max_tokens": 1024}},
                ),
                Step(
                    name="feedback-qwen-32b",
                    task="feedback",
                    model_name="RedHatAI/Qwen3-32B-FP8-dynamic",
                    file_directory=ROOT.joinpath("instructions/feedback"),
                    post_field_renamer={k: f"{k}_qwen" for k in FEEDBACK_FIELDS},
                    anno_init_kwargs={"max_model_len": 3000, "max_num_batched_tokens": 8192, "max_num_seqs": 96},
                    anno_run_kwargs={"sampling_params": {**SAMPLING_PARAMS["qwen3"], "max_tokens": 1024}},
                ),
                Step(
                    name="revisor-gemma-27b",
                    task="revisor",
                    model_name="RedHatAI/gemma-3-27b-it-FP8-dynamic",
                    file_directory=ROOT.joinpath("instructions/revisor"),
                    post_field_renamer={"nieuwe-definitie": "definitie-v2"},
                    anno_init_kwargs={"max_model_len": 3000},
                    anno_run_kwargs={"sampling_params": {**SAMPLING_PARAMS["gemma3"], "max_tokens": 480}},
                ),
            ]
        )

    steps.extend(
        [
            Step(
                name="judge-gemma-27b",
                task="judge",
                model_name="RedHatAI/gemma-3-27b-it-FP8-dynamic",
                file_directory=ROOT.joinpath(
                    "instructions/judge-after-revision" if add_revision_stage else "instructions/judge"
                ),
                post_field_renamer={k: f"{k}_judge_gemma" for k in JUDGE_FIELDS},
                anno_init_kwargs={"max_model_len": 3200},
                anno_run_kwargs={"sampling_params": {**SAMPLING_PARAMS["gemma3"], "max_tokens": 1024}},
            ),
            Step(
                name="ranker-gemma-27b",
                task="ranker",
                model_name="RedHatAI/gemma-3-27b-it-FP8-dynamic",
                file_directory=ROOT.joinpath(
                    "instructions/ranker-after-revision" if add_revision_stage else "instructions/ranker"
                ),
                post_field_renamer={k: f"{k}_ranker_gemma" for k in RANKER_FIELDS},
                anno_init_kwargs={"max_model_len": 3200},
                anno_run_kwargs={"sampling_params": {**SAMPLING_PARAMS["gemma3"], "max_tokens": 768}},
            ),
            Step(
                name="judge-qwen-32b",
                task="judge",
                model_name="RedHatAI/Qwen3-32B-FP8-dynamic",
                file_directory=ROOT.joinpath(
                    "instructions/judge-after-revision" if add_revision_stage else "instructions/judge"
                ),
                post_field_renamer={k: f"{k}_judge_qwen" for k in JUDGE_FIELDS},
                anno_init_kwargs={"max_model_len": 3200, "max_num_batched_tokens": 8192, "max_num_seqs": 16},
                anno_run_kwargs={"sampling_params": {**SAMPLING_PARAMS["qwen3"], "max_tokens": 1024}},
            ),
            Step(
                name="ranker-qwen-32b",
                task="ranker",
                model_name="RedHatAI/Qwen3-32B-FP8-dynamic",
                file_directory=ROOT.joinpath(
                    "instructions/ranker-after-revision" if add_revision_stage else "instructions/ranker"
                ),
                post_field_renamer={k: f"{k}_ranker_qwen" for k in RANKER_FIELDS},
                anno_init_kwargs={"max_model_len": 3200, "max_num_batched_tokens": 8192, "max_num_seqs": 16},
                anno_run_kwargs={"sampling_params": {**SAMPLING_PARAMS["qwen3"], "max_tokens": 768}},
            ),
        ]
    )

    root_pdout.mkdir(parents=True, exist_ok=True)

    pipe = Pipeline(steps)
    # Save pipe config
    with root_pdout.joinpath("pipeline.json").open("w", encoding="utf-8") as fh:
        data = pipe.to_dict()
        data = {
            **data,
            "num_samples": num_samples,
            "min_num_examples": min_num_examples,
            "default_max_num_seqs": default_max_num_seqs,
            "defaul_max_model_len": defaul_max_model_len,
            "default_max_tokens": default_max_tokens,
            "default_num_batched_tokens": default_num_batched_tokens,
            "greedy": greedy,
            "gpu_memory_utilization": gpu_memory_utilization,
            "num_proc": num_proc,
            "root_pdout": str(root_pdout),
            "new_hub_id": new_hub_id,
            "add_revision_stage": add_revision_stage,
        }
        json.dump(data, fh, indent=4, cls=PathCompatibleEncoder)

    feedback_cols = {}
    prev_model_name = None
    anno = None

    failed_samples_fh = root_pdout.joinpath("failed_samples.jsonl").open("a", encoding="utf-8")

    no_revision_ds = None
    for step in pipe:
        print(f"Step {step.idx + 1}: {step.name} ({step.task}) using model {step.model_name}")

        if step.task == "feedback":
            feedback_cols[step.name] = list(step.post_field_renamer.values())

        pdout = root_pdout.joinpath(f"{step.idx + 1}-{step.task}-{step.name}")

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
                # We're past feedback and we have columns that indicate (no) revision is needed
                # so we can split off the 'no revision needed' samples to save compute time
                if step.task != "feedback" and any("revisie-nodig" in c for c in ds.column_names):
                    new_no_revision_ds = ds.filter(
                        lambda feedback: not bool(feedback),
                        input_columns=["feedback"],
                        num_proc=num_proc,
                    )
                    if no_revision_ds is None:
                        no_revision_ds = new_no_revision_ds
                    else:
                        no_revision_ds = concatenate_datasets([no_revision_ds, new_no_revision_ds])

                    ds = ds.filter(lambda feedback: bool(feedback), input_columns=["feedback"], num_proc=num_proc)
                continue

        pdout_done.mkdir(parents=True, exist_ok=True)

        task_prefix = f"{step.idx + 1}-{step.name}_"

        if step.model_name != prev_model_name:
            # Reinitialize annotator if model changes
            default_init_kwargs = {
                "model": step.model_name,
                "num_proc": num_proc,
                "max_model_len": defaul_max_model_len,
                "max_num_seqs": default_max_num_seqs,
                "tensor_parallel_size": torch.cuda.device_count(),
                "gpu_memory_utilization": gpu_memory_utilization,
                "verbose": True,
                "max_num_batched_tokens": default_num_batched_tokens,
            }
            anno = Annotator(**{**default_init_kwargs, **step.anno_init_kwargs})
            prev_model_name = step.model_name

        if greedy:
            if "sampling_params" in step.anno_run_kwargs and "temperature" in step.anno_run_kwargs["sampling_params"]:
                print(
                    "Warning: Overriding provided temperature in anno_run_kwargs.sampling_params"
                    " to 0.0 due to greedy=True."
                )
            sampling_params = {"max_tokens": default_max_tokens, "temperature": 0.0}
        else:
            sampling_params = {"max_tokens": default_max_tokens}

        if "sampling_params" in step.anno_run_kwargs:
            sampling_params = {**sampling_params, **step.anno_run_kwargs["sampling_params"]}

        if step.task == "ranker" and add_revision_stage:
            plaatsen = ["eerste-plaats", "tweede-plaats", "derde-plaats"]
            plaatsen_cols = [step.post_field_renamer[pos] for pos in plaatsen]
            validate_fn = partial(check_rank_validity, plaatsen_cols=plaatsen_cols)
        else:
            validate_fn = None

        default_run_kwargs = {
            "output_dir": pdout,
            "dataset": ds,
            "full_prompt_template": step.full_prompt_template,
            "prompt_template_prefix": step.prompt_template_prefix,
            "output_schema": step.schema,
            "keep_columns": True,
            "sampling_params": sampling_params,
            "task_prefix": task_prefix,
            "sort_by_length": True,
            "num_retries_invalid": 3,
            "validate_fn": validate_fn,
        }
        ds = anno.annotate_dataset(**{**default_run_kwargs, **step.anno_run_kwargs})

        # Only keep successful annotations
        num_before = len(ds)
        filtered_ds = ds.filter(
            lambda is_valid: is_valid, input_columns=[f"{task_prefix}valid_fields"], num_proc=num_proc
        )
        num_after = len(filtered_ds)
        if num_after == 0:
            raise ValueError(f"All samples were filtered out after {step.name} with model {step.model_name}!")
        elif num_after < num_before:
            invalid_ds = ds.filter(
                lambda is_valid: not is_valid, input_columns=[f"{task_prefix}valid_fields"], num_proc=num_proc
            )
            print(
                f"{num_before - num_after} samples were invalid after {step.name} with model {step.model_name}."
                f" Dataset now has {num_after} samples remaining."
            )
            print(
                "A potential cause is that the model could not respond with the full schema due to"
                " 'sampling_params.max_tokens' being too low. The stopping reasons for the invalid"
                " samples are given below (valid='stop')."
            )
            counts = Counter(invalid_ds[f"{task_prefix}finish_reason"])
            counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
            for reason, count in counts.items():
                print(f"- {reason}: {count:,}")
            for sample in invalid_ds.to_list():
                sample["invalid_reason"] = f"{step.idx + 1}-{step.name}: Did not produce valid_fields=True"
                failed_samples_fh.write(f"{json.dumps(sample)}\n")
            del invalid_ds

        # Re-assign to `ds` for next step
        ds = filtered_ds.rename_columns(step.post_field_renamer)

        if step.task == "feedback":
            if step.last_of_task:
                ds = ds.map(
                    concat_feedback,
                    num_proc=num_proc,
                    fn_kwargs={"feedback_cols": feedback_cols},
                )
                feedback_cols = {}
        elif step.task == "judge":
            # These dataset fields are a dictionary of "analyse" and "score" and should be split
            # into separate fields
            cols_to_split = list(step.post_field_renamer.values())
            for col in cols_to_split:
                ds = ds.map(
                    lambda x: {f"analyse_{col}": x[col]["analyse"], f"score_{col}": x[col]["score"]},
                    num_proc=num_proc,
                    remove_columns=[col],
                )
        elif step.task == "revisor" or (step.task == "definer" and "revisor" not in pipe):
            if step.last_of_task and "ranker" in pipe:
                # After revision, if there is a ranker, we want to compare all three definitions (v1, v2, original)
                # To do so, we randomly shuffle them and add them to separate columns `random-definitie-1/2/3`
                # and store the order in a separate column called "shuffle-order" with values like:
                # ["definitie-v1", "definitie-v2", "originele-definitie"]
                ds = ds.map(shuffle_definitions, num_proc=num_proc)

        # Add new columns to the old 'already good' feedback ds
        if no_revision_ds is not None:
            new_cols = set(ds.column_names) - set(no_revision_ds.column_names)
            for col in new_cols:
                no_revision_ds = no_revision_ds.add_column(col, [None] * len(no_revision_ds))

        print(f"Done step {step.idx + 1}: {step.name}, saving to {pdout_done}")
        ds.save_to_disk(pdout_done)

    failed_samples_fh.close()
    if no_revision_ds is not None:
        print(f"Merging back {len(no_revision_ds)} samples that did not require revision.")
        ds = concatenate_datasets([ds, no_revision_ds])

    ds.save_to_disk(root_pdout.joinpath("full_process_done"))

    if new_hub_id:
        if "{num_samples}" in new_hub_id:
            new_hub_id = new_hub_id.format(num_samples=len(ds))
        print(f"Pushing final dataset to the hub at {new_hub_id}")
        ds.push_to_hub(new_hub_id, private=True)


if __name__ == "__main__":
    main(
        num_samples=None,
        add_revision_stage=False,
        greedy=False,
        root_pdout=ROOT.joinpath("outputs/anw_no_revision"),
    )
