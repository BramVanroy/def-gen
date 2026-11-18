"""Configuration and execution of the definition generation workflow."""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from collections import Counter
from functools import partial
from pathlib import Path
from random import Random
from typing import Any, Literal

import numpy as np
from datasets import Value, concatenate_datasets, load_dataset
from llm_annotator import Annotator
from llm_annotator.clients import Client, ProviderRuntimeOptions
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from def_gen.pipeline import Pipeline, Step
from def_gen.utils import ROOT, PathCompatibleEncoder


JUDGE_FIELDS = [
    "begrijpelijkheid",
    "betekenis",
    "gebruik",
    "structuur",
    "substitueerbaarheid",
    "correctheid",
]
FEEDBACK_FIELDS = [*JUDGE_FIELDS, "revisie-nodig"]
REVISED_FIELD = "nieuwe-definitie"

DEFAULT_DATA_PATH = ROOT.joinpath("data/anw_lemma_definitions.jsonl")

CLIENT_TYPE_CHOICES = ["openai", "anthropic", "vllm", "vllm-offline"]


class PipelineStepConfig(BaseModel):
    """User-provided pipeline step configuration.

    Args:
        name: Unique step name.
        task: Pipeline task type.
        model_name: Model identifier.
        provider: Client backend provider (openai, anthropic, vllm,
            vllm-offline). Defaults to vllm-offline when omitted.
        file_directory: Instruction directory with shared/specific/schema files.
        full_prompt_template: Optional direct prompt template content.
        prompt_template_prefix: Optional prompt prefix when using direct
            templates.
        output_schema: Optional output schema when not using schema file.
        system_message: Optional system message passed to the annotator.
        client_init_kwargs: Extra kwargs forwarded to the client constructor.
        client_run_kwargs: Runtime options kwargs for the provider.
        post_field_renamer: Mapping from output fields to destination names.
            Judge and feedback steps default to `{field}_{name}`.
        definition_column: Column that a judge step evaluates. Required for
            judge steps and rejected for other tasks.
        order_rotation: Ranker steps only. The shuffled candidate order
            (`shuffle-order`) is rotated left by this many places before it
            is shown as `positie-definitie-i`. Running rotations 0 to n-1 in
            separate ranker steps gives a cyclic Latin square over the
            positions. Must be smaller than the number of candidates.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str
    task: Literal["definer", "feedback", "revisor", "judge", "ranker"]
    model_name: str
    provider: Literal["openai", "anthropic", "vllm", "vllm-offline"] | None = (
        None
    )
    file_directory: Path | None = None
    full_prompt_template: str | None = None
    prompt_template_prefix: str = ""
    output_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    system_message: str | None = None
    client_init_kwargs: dict[str, Any] = Field(default_factory=dict)
    client_run_kwargs: dict[str, Any] = Field(default_factory=dict)
    post_field_renamer: dict[str, str] = Field(default_factory=dict)
    definition_column: str | None = None
    order_rotation: int = Field(default=0, ge=0)

    @field_validator("file_directory", mode="before")
    @classmethod
    def _coerce_file_directory(cls, value: str | Path | None) -> Path | None:
        """Normalize file directory values to Path objects."""
        if value is None:
            return None
        return Path(value)

    @model_validator(mode="after")
    def _validate_prompt_source(self) -> PipelineStepConfig:
        """Require either an instruction directory or explicit template."""
        if self.file_directory is None and self.full_prompt_template is None:
            raise ValueError(
                "Each pipeline step needs either file_directory or"
                " full_prompt_template."
            )
        return self

    def to_step(self) -> Step:
        """Convert this config into a pipeline Step object."""
        return Step(
            name=self.name,
            task=self.task,
            model_name=self.model_name,
            provider=self.provider,
            file_directory=self.file_directory,
            full_prompt_template=self.full_prompt_template,
            prompt_template_prefix=self.prompt_template_prefix,
            schema=self.output_schema,
            system_message=self.system_message,
            post_field_renamer=self.post_field_renamer,
            client_init_kwargs=self.client_init_kwargs,
            client_run_kwargs=self.client_run_kwargs,
            definition_column=self.definition_column,
            order_rotation=self.order_rotation,
        )


class GenerateDefinitionsConfig(BaseModel):
    """Configuration for the definitions generation workflow.

    Args:
        from_hub_id: Optional hub dataset id to initialize from instead of
            the source data file.
        num_samples: Number of examples to keep, taken as the first rows of
            the source data file or of the hub dataset. Use None for all rows.
        min_num_examples: Minimum number of examples per entry.
        overwrite: Whether to delete existing step outputs before running.
        batch_size: Number of requests per annotation batch.
        num_proc: Number of workers for dataset operations.
        root_output_dir: Output directory for checkpoints and final dataset.
        hub_id: Optional hub dataset id to push final output.
        workflow: Workflow preset for whether revision is included.
        steps: Optional explicit pipeline steps.
        seed: Random seed used for deterministic data transforms.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    from_hub_id: str | None = None
    num_samples: int | None = 512
    min_num_examples: int = Field(default=3, ge=1)
    overwrite: bool = False
    batch_size: int = Field(default=256, ge=1)
    num_proc: int = Field(default=4, ge=1)
    root_output_dir: Path = ROOT.joinpath("outputs")
    hub_id: str | None = "BramVanroy/anw-defgen-{num_samples}"
    steps: list[PipelineStepConfig] | None = None
    seed: int = 42

    @field_validator("root_output_dir", mode="before")
    @classmethod
    def _coerce_path(cls, value: str | Path) -> Path:
        """Normalize output directory values to Path objects."""
        return Path(value)

    @classmethod
    def from_sources(
        cls,
        config_data: dict[str, Any] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> GenerateDefinitionsConfig:
        """Build config from optional file data and CLI overrides."""
        merged: dict[str, Any] = {}
        if config_data:
            merged.update(config_data)
        if overrides:
            merged.update(overrides)
        return cls.model_validate(merged)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config to a dict for JSON output."""
        return self.model_dump(mode="python")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for deterministic behavior across libraries."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _create_client_and_runtime_options(
    step: Step,
) -> tuple[Client, ProviderRuntimeOptions]:
    """Create the configured llm-annotator client for a step."""
    provider = step.provider

    try:
        if provider == "openai":
            from llm_annotator import OpenAIClient, OpenAIRuntimeOptions

            return OpenAIClient(
                model=step.model_name, **step.client_init_kwargs
            ), OpenAIRuntimeOptions(**step.client_run_kwargs)
        elif provider == "anthropic":
            from llm_annotator import ClaudeClient, ClaudeRuntimeOptions

            return ClaudeClient(
                model=step.model_name, **step.client_init_kwargs
            ), ClaudeRuntimeOptions(**step.client_run_kwargs)
        elif provider == "vllm":
            from llm_annotator import (
                VLLMOnlineClient,
                VLLMOnlineRuntimeOptions,
            )

            return VLLMOnlineClient(
                model=step.model_name, **step.client_init_kwargs
            ), VLLMOnlineRuntimeOptions(**step.client_run_kwargs)
        elif provider == "vllm-offline":
            from llm_annotator import (
                VLLMOfflineClient,
                VLLMOfflineRuntimeOptions,
            )

            return VLLMOfflineClient(
                model=step.model_name, **step.client_init_kwargs
            ), VLLMOfflineRuntimeOptions(**step.client_run_kwargs)
        else:
            raise ValueError(
                f"Unsupported provider {provider!r} for client creation."
            )
    except ImportError as import_exc:
        if provider == "openai":
            raise ImportError(
                "OpenAI client dependencies are missing. Install with"
                " `uv sync --extra openai`."
            ) from import_exc
        if provider == "anthropic":
            raise ImportError(
                "Anthropic client dependencies are missing. Install with"
                " `uv sync --extra anthropic`."
            ) from import_exc
        raise ImportError(
            "vLLM client dependencies are missing. Install with"
            " `uv sync --extra vllm`."
        ) from import_exc


def concat_feedback(
    sample: dict[str, Any],
    feedback_cols: dict[str, list[str]],
) -> dict[str, str]:
    """Combine reviewer feedback columns into a single field."""
    feedback = ""
    for reviewer_idx, (_, cols) in enumerate(feedback_cols.items(), 1):
        revisie_name = next(
            (c for c in cols if c.startswith("revisie-nodig")), None
        )
        if revisie_name is not None and not sample[revisie_name]:
            continue

        reviewer_feedback = f"*Feedback van reviewer {reviewer_idx}*\n"
        reviewer_points = []
        for col in cols:
            if (
                col in sample
                and "revisie-nodig" not in col
                and sample[col] is not None
            ):
                colname = next(
                    (f for f in FEEDBACK_FIELDS if col.startswith(f)), col
                ).lower()
                if bool(re.search(r"([^.])\1{2,}", sample[col])):
                    print(
                        "  - Skipping feedback for column"
                        f" {col} due to repeated characters: {sample[col]}"
                    )
                    continue
                reviewer_points.append(f"- {colname}: {sample[col]}")

        if reviewer_points:
            feedback += reviewer_feedback + "\n".join(reviewer_points) + "\n\n"

    if feedback == "":
        print(
            f"Warning: No feedback generated for lemmavorm {sample['lemmavorm']!r}"
        )

    return {"feedback": feedback.strip()}


def prepare_examples(
    voorbeelden: list[str],
    word: str,
    num_examples: int = 3,
    context_words: int = 50,
) -> dict[str, str]:
    """Format example sentences around the target word."""
    voorbeelden_str = ""
    lc_word = word.lower()
    Random(42).shuffle(voorbeelden)

    for vb in voorbeelden[:num_examples]:
        vb = " ".join(vb.strip().split())
        vb_words = vb.split()
        vb_lc_words = vb.lower().split()
        word_idx = next(
            (i for i, w in enumerate(vb_lc_words) if lc_word in w), -1
        )
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


def shuffle_definitions(sample: dict[str, Any]) -> dict[str, Any]:
    """Shuffle candidate definitions deterministically for ranking."""
    if "definitie-v2" in sample:
        defs = [
            sample["definitie-v1"],
            sample["definitie-v2"],
            sample["originele-definitie"],
        ]
        names = ["definitie-v1", "definitie-v2", "originele-definitie"]
    else:
        defs = [sample["definitie-v1"], sample["originele-definitie"]]
        names = ["definitie-v1", "originele-definitie"]
    combined = list(zip(defs, names, strict=True))

    seed = int(hashlib.sha256(str(sample).encode("utf-8")).hexdigest(), 16) % (
        2**32
    )
    Random(seed).shuffle(combined)
    defs, names = zip(*combined, strict=True)
    output = {
        "random-definitie-1": defs[0],
        "random-definitie-2": defs[1],
    }
    if len(defs) == 3:
        output["random-definitie-3"] = defs[2]

    output["shuffle-order"] = list(names)
    return output


def limit_samples(ds: Any, num_samples: int | None) -> Any:
    """Keep the first `num_samples` rows.

    Args:
        ds: Dataset to limit.
        num_samples: Number of rows to keep. None or a non-positive value
            keeps all rows.

    Returns:
        The limited dataset.
    """
    if num_samples is not None and 0 < num_samples < len(ds):
        return ds.select(range(num_samples))
    return ds


def init_data(
    num_samples: int | None = 8,
    min_num_examples: int = 3,
    num_proc: int | None = None,
    data_path: Path | None = DEFAULT_DATA_PATH,
) -> Any:
    """Load and preprocess the ANW source dataset."""
    ds = load_dataset(
        "json",
        data_files=[str(data_path)],
    )["train"]

    ds = (
        ds.map(
            lambda voorbeelden, word: {
                # Only keep examples that contain the target word to ensure relevance for the model prompt
                "voorbeelden": [
                    vb for vb in voorbeelden if word.lower() in vb.lower()
                ]
            },
            input_columns=["voorbeelden", "lemmavorm"],
            num_proc=num_proc,
        )
        .filter(
            lambda x: (
                x["woordsoort"] == "substantief"
                and len(x["definitie"].split()) > 5
                and x["lemmavorm"][0].isalpha()
                and x["lemmavorm"][0].islower()
                and not x["artikel_meerdere_betekenissen"]
                and "ook:"
                not in x["definitie"].lower()  # Also counts as polysemic
                and len(x["voorbeelden"]) >= min_num_examples
            ),
            num_proc=num_proc,
        )
        .remove_columns(
            [
                "mini_definitie",
                "woordsoort",
                "artikel_id",
                "artikel_meerdere_betekenissen",
                "betekenis_id",
            ]
        )
        .rename_column("definitie", "originele-definitie")
        .map(
            prepare_examples,
            input_columns=["voorbeelden", "lemmavorm"],
            num_proc=num_proc,
            fn_kwargs={"num_examples": min_num_examples},
        )
        .shuffle(seed=42)
        .cast_column("artikel_laatst_bewerkt", Value("string"))
    )

    return limit_samples(ds, num_samples)


def check_rank_validity(
    sample: dict[str, Any], plaatsen_cols: list[str]
) -> bool:
    """Return whether all three ranking choices appear exactly once."""
    if len(plaatsen_cols) != 3:
        raise ValueError(
            "Expected 3 rank positions for validity check"
            " (eerste, tweede, derde plaats)."
        )

    rank_values = {"Definitie 1", "Definitie 2", "Definitie 3"}
    for col_name in plaatsen_cols:
        col_value = sample[col_name]
        if col_value is None:
            continue
        if col_value not in rank_values:
            return False
        rank_values.remove(col_value)

    return len(rank_values) == 0


def build_pipeline(config: GenerateDefinitionsConfig) -> Pipeline:
    """Build a pipeline for the selected workflow configuration."""
    if config.steps:
        return Pipeline(
            [step_config.to_step() for step_config in config.steps]
        )
    else:
        raise NotImplementedError(
            "Currently only explicit steps configuration is supported."
        )


ROW_IDX_COL = "__row_idx"
JUDGE_INPUT_COL = "te-beoordelen-definitie"
POSITION_PREFIX = "positie-definitie-"


def _drop_invalid(
    ds: Any, step: Step, step_dir: Path, task_prefix: str, num_proc: int
) -> Any:
    """Remove rows with `valid_fields=False` and record the attrition.

    Writes `attrition.json` and `failed_samples.jsonl` into `step_dir`. A rerun
    of the step overwrites both files.

    Args:
        ds: Annotated dataset with the `valid_fields` and `finish_reason` columns.
        step: The step that produced `ds`.
        step_dir: Output directory of the step.
        task_prefix: Column prefix of the step.
        num_proc: Number of workers for dataset operations.

    Returns:
        The dataset with valid rows only.

    Raises:
        ValueError: If no row is valid.
    """
    valid_col = f"{task_prefix}valid_fields"
    valid_ds = ds.filter(
        lambda is_valid: is_valid, input_columns=[valid_col], num_proc=num_proc
    )
    invalid_ds = ds.filter(
        lambda is_valid: not is_valid,
        input_columns=[valid_col],
        num_proc=num_proc,
    )
    counts = dict(
        Counter(invalid_ds[f"{task_prefix}finish_reason"]).most_common()
    )

    step_dir.mkdir(parents=True, exist_ok=True)
    step_dir.joinpath("attrition.json").write_text(
        json.dumps(
            {
                "n_in": len(ds),
                "n_valid": len(valid_ds),
                "n_invalid": len(invalid_ds),
                "invalid_finish_reasons": counts,
            },
            indent=4,
        ),
        encoding="utf-8",
    )
    with step_dir.joinpath("failed_samples.jsonl").open(
        "w", encoding="utf-8"
    ) as fh:
        for sample in invalid_ds.to_list():
            sample["invalid_reason"] = (
                f"{step.idx + 1}-{step.name}: Did not produce"
                " valid_fields=True"
            )
            fh.write(f"{json.dumps(sample, default=str)}\n")

    if len(valid_ds) == 0:
        raise ValueError(
            "All samples were filtered out after"
            f" {step.name} with model {step.model_name}!"
        )
    if len(invalid_ds) > 0:
        print(
            f"{len(invalid_ds)} samples were invalid after"
            f" {step.name} with model {step.model_name}."
            f" Dataset now has {len(valid_ds)} samples remaining."
        )
        print(
            "A potential cause is that the model could not respond with"
            " the full schema due to too-low max_completion_tokens in"
            " client_run_kwargs. The stopping reasons for invalid"
            " samples are below (valid='stop')."
        )
        for reason, count in counts.items():
            print(f"- {reason}: {count:,}")

    return valid_ds


def _merge_unrevised(
    revised: Any, unrevised: Any, step: Step, task_prefix: str
) -> Any:
    """Combine revised rows with rows that had no feedback, in original order.

    Unrevised rows get the original definition as revised definition, None
    for the other revisor outputs, and `revisie-uitgevoerd=False`.

    Args:
        revised: Rows that were sent to the model (possibly empty).
        unrevised: Rows with empty feedback.
        step: The revisor step.
        task_prefix: Column prefix of the step.

    Returns:
        One dataset without the temporary row index column.
    """
    renamer = step.post_field_renamer
    fields = list(step.schema["properties"]) if step.schema else list(renamer)
    new_def_col = renamer.get(REVISED_FIELD, REVISED_FIELD)

    unrevised = unrevised.add_column(
        "revisie-uitgevoerd", [False] * len(unrevised)
    )
    unrevised = unrevised.add_column(new_def_col, unrevised["definitie-v1"])
    for field_name in fields:
        col = renamer.get(field_name, field_name)
        if col != new_def_col:
            unrevised = unrevised.add_column(col, [None] * len(unrevised))

    if len(revised) == 0:
        merged = unrevised
    else:
        revised = revised.add_column(
            "revisie-uitgevoerd", [True] * len(revised)
        )
        for col in revised.column_names:
            if col not in unrevised.column_names:
                fill = True if col == f"{task_prefix}valid_fields" else None
                unrevised = unrevised.add_column(col, [fill] * len(unrevised))
        unrevised = unrevised.select_columns(revised.column_names).cast(
            revised.features
        )
        merged = concatenate_datasets([revised, unrevised])

    return merged.sort(ROW_IDX_COL).remove_columns([ROW_IDX_COL])


def _add_ranker_positions(ds: Any, step: Step, num_proc: int) -> Any:
    """Fill `positie-definitie-i` columns from `shuffle-order` rotated left.

    Position i shows the candidate `rotated[i - 1]`. The used order is stored
    in `volgorde_<step.name>`.

    Args:
        ds: Dataset with `shuffle-order` and the candidate columns.
        step: The ranker step.
        num_proc: Number of workers for dataset operations.

    Returns:
        The dataset with the position columns and the order column.

    Raises:
        ValueError: If the rotation is not smaller than the number of
            candidates.
    """
    num_candidates = len(ds[0]["shuffle-order"])
    if not 0 <= step.order_rotation < num_candidates:
        raise ValueError(
            f"order_rotation {step.order_rotation} of step {step.name!r}"
            f" must be in [0, {num_candidates})."
        )

    def add_positions(sample: dict[str, Any]) -> dict[str, Any]:
        order = sample["shuffle-order"]
        rotated = order[step.order_rotation :] + order[: step.order_rotation]
        out = {
            f"{POSITION_PREFIX}{i}": sample[source]
            for i, source in enumerate(rotated, 1)
        }
        out[f"volgorde_{step.name}"] = rotated
        return out

    return ds.map(add_positions, num_proc=num_proc)


def run_generation(config: GenerateDefinitionsConfig) -> None:
    """Run the full generation workflow for the provided config."""
    set_seed(config.seed)

    if config.from_hub_id:
        ds = limit_samples(
            load_dataset(config.from_hub_id)["train"], config.num_samples
        )
    else:
        ds = init_data(
            num_samples=config.num_samples,
            min_num_examples=config.min_num_examples,
            num_proc=config.num_proc,
        )
    pipe = build_pipeline(config)

    config.root_output_dir.mkdir(parents=True, exist_ok=True)
    with config.root_output_dir.joinpath("pipeline.json").open(
        "w", encoding="utf-8"
    ) as fh:
        data = {**pipe.to_dict(), **config.to_dict()}
        json.dump(data, fh, indent=4, cls=PathCompatibleEncoder)

    feedback_cols: dict[str, list[str]] = {}
    prev_client_key = None
    anno = None

    for step in pipe:
        print(
            f"Step {step.idx + 1}: {step.name} ({step.task}) using model {step.model_name}"
        )

        if step.task == "feedback":
            feedback_cols[step.name] = list(step.post_field_renamer.values())

        step_dir = config.root_output_dir.joinpath(
            f"{step.idx + 1}-{step.task}-{step.name}"
        )

        if config.overwrite and step_dir.is_dir():
            shutil.rmtree(step_dir)

        step_done_dir = step_dir / "done"

        if step_done_dir.is_dir() and any(step_done_dir.iterdir()):
            try:
                ds = ds.load_from_disk(step_done_dir)
            except Exception:
                shutil.rmtree(step_done_dir)
            else:
                print(
                    "  - Skipping step because output dir"
                    f" {step_done_dir} already exists and is non-empty."
                )
                continue

        step_done_dir.mkdir(parents=True, exist_ok=True)

        task_prefix = f"{step.idx + 1}-{step.name}_"

        client_key = (
            step.model_name,
            json.dumps(step.client_init_kwargs, sort_keys=True, default=str),
        )
        if client_key != prev_client_key:
            if anno is not None:
                anno.destroy()
            client, options = _create_client_and_runtime_options(step)
            anno = Annotator(
                client=client,
                batch_size=config.batch_size,
                num_proc=config.num_proc,
                verbose=True,
            )
            prev_client_key = client_key

        if step.task == "ranker" and "revisor" in pipe:
            plaatsen = ["eerste-plaats", "tweede-plaats", "derde-plaats"]
            plaatsen_cols = [step.post_field_renamer[pos] for pos in plaatsen]
            validate_fn = partial(
                check_rank_validity, plaatsen_cols=plaatsen_cols
            )
        else:
            validate_fn = None

        unrevised_ds = None
        if step.task == "revisor":
            ds = ds.add_column(ROW_IDX_COL, list(range(len(ds))))
            unrevised_ds = ds.filter(
                lambda feedback: not bool(feedback),
                input_columns=["feedback"],
                num_proc=config.num_proc,
            )
            ds = ds.filter(
                lambda feedback: bool(feedback),
                input_columns=["feedback"],
                num_proc=config.num_proc,
            )

        elif step.task == "judge":
            ds = ds.add_column(JUDGE_INPUT_COL, ds[step.definition_column])
        elif step.task == "ranker":
            ds = _add_ranker_positions(ds, step, config.num_proc)

        if len(ds) > 0:
            ds = anno.annotate_dataset(
                output_dir=step_dir,
                dataset=ds,
                prompt_template=step.full_prompt_template,
                output_schema=step.schema,
                keep_columns=True,
                options=options,
                task_prefix=task_prefix,
                sort_by_length=True,
                num_retries_invalid=3,
                validate_fn=validate_fn,
                system_message=step.system_message,
            )
            ds = _drop_invalid(
                ds, step, step_dir, task_prefix, config.num_proc
            )
            ds = ds.rename_columns(step.post_field_renamer)

        if step.task == "revisor":
            ds = _merge_unrevised(ds, unrevised_ds, step, task_prefix)
        elif step.task == "judge":
            ds = ds.remove_columns([JUDGE_INPUT_COL])
        elif step.task == "ranker":
            ds = ds.remove_columns(
                [c for c in ds.column_names if c.startswith(POSITION_PREFIX)]
            )

        if step.task == "feedback" and step.last_of_task:
            ds = ds.map(
                concat_feedback,
                num_proc=config.num_proc,
                fn_kwargs={"feedback_cols": feedback_cols},
            )
            feedback_cols = {}
        elif step.task == "judge":
            cols_to_split = list(step.post_field_renamer.values())
            for col in cols_to_split:
                ds = ds.map(
                    lambda x, col=col: {
                        f"analyse_{col}": x[col]["analyse"],
                        f"score_{col}": x[col]["score"],
                    },
                    num_proc=config.num_proc,
                    remove_columns=[col],
                )
        elif (
            step.task == "revisor"
            or (step.task == "definer" and "revisor" not in pipe)
        ) and (step.last_of_task and "ranker" in pipe):
            ds = ds.map(shuffle_definitions, num_proc=config.num_proc)

        print(
            f"Done step {step.idx + 1}: {step.name}, saving to {step_done_dir}"
        )
        ds.save_to_disk(step_done_dir)

    if anno is not None:
        anno.destroy()

    ds.save_to_disk(config.root_output_dir.joinpath("full_process_done"))

    if config.hub_id:
        new_hub_id = config.hub_id
        if "{num_samples}" in new_hub_id:
            new_hub_id = new_hub_id.format(num_samples=len(ds))
        print(f"Pushing final dataset to the hub at {new_hub_id}")
        ds.push_to_hub(new_hub_id, private=True)
