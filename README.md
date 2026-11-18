# def-gen

[![CI](https://github.com/BramVanroy/def-gen/actions/workflows/ci.yml/badge.svg)](https://github.com/BramVanroy/def-gen/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/BramVanroy/def-gen/branch/main/graph/badge.svg)](https://codecov.io/gh/BramVanroy/def-gen)
[![License](https://img.shields.io/github/license/BramVanroy/def-gen)](LICENSE)

Generating and evaluating dictionary definitions with LLMs.

This project studies how definitions that open-weight LLMs write for Dutch
nouns compare with the definitions of the ANW (Algemeen Nederlands
Woordenboek). A writer model gets a lemma and example sentences from the ANW
and writes a definition. Open-weight LLM judges score the written definition
and the ANW definition for the same lemma and examples, and rank the two in
pairs. Writers are also ranked against each other. The code runs these steps,
merges the outputs and computes the statistics (win rates, agreement, biases).
No commercial API is needed. Steps for paid providers exist but are refused by
default.

## Installation

```bash
git clone https://github.com/BramVanroy/def-gen.git
cd def-gen
uv sync --all-extras
uv run pre-commit install
```

The extras select the LLM provider: `vllm`, `openai` or `anthropic`. For
example, `uv sync --extra openai` installs only what the OpenAI client needs.

## Data

The input is the ANW XML dump. How to obtain it and under which licence it may
be used is not covered here.

```bash
uv run python scripts/extract_anw_articles.py data/anw.xml
```

This writes `data/anw_lemma_articles.jsonl` and
`data/anw_lemma_definitions.jsonl` (one row per meaning). When a config has no
`from_hub_id`, `init_data` reads `data/anw_lemma_definitions.jsonl` and keeps a
row if:

- the word class is noun (`substantief`),
- the article has one meaning and the definition does not contain "ook:",
- the ANW definition has more than 5 words,
- the lemma starts with a lowercase letter,
- at least `min_num_examples` of its examples contain the lemma (as a
  lowercase substring, so "huis" matches "huisdier").

The first `min_num_examples` of the matching examples (after a seeded shuffle)
are shown to the models, each cut to 50 words of context on both sides of the
lemma.

The study uses the item set
`BramVanroy/anw-15-4-2026-monosemous-nouns-5examples`, which
`scripts/upload_prepped_data.py` builds with `min_num_examples=5` and pushes as
a private Hub dataset. `scripts/make_item_split.py` freezes a split of it
(seed 42) in `data/splits/anw-5ex`:

- `dev`: 200 items for prompt work.
- `eval`: 4,750 items for the study.

```bash
uv run python scripts/make_item_split.py
```

## Pipeline

A run is a JSON config with a list of steps (see `config/study/`). Each step
has a task, a model, a provider (`vllm-offline` by default) and an instruction
directory in `instructions/` with Dutch prompts (`shared.md`, `specific.md`)
and an output schema (`schema.json`). The steps run in the listed order and
each adds columns to the dataset.

| Task | Reads | Writes |
| --- | --- | --- |
| `definer` | `lemmavorm`, `voorbeelden` | `definitie` (renamed to `definitie-v1`) |
| `feedback` | `definitie-v1` | six criterion texts and `revisie-nodig` per step, joined in `feedback` |
| `revisor` | `definitie-v1`, `feedback` | `nieuwe-definitie` (renamed, for example `definitie-v2`) |
| `judge` | the column in `definition_column` | `analyse_<column>` and `score_<column>` per criterion |
| `ranker` | two or three candidate definitions | `argumentatie_*`, `ranking_*` and `volgorde_<step>` |

A `definer` step has to come first and only one is allowed. A `revisor` needs
an earlier `feedback` step. A pipeline of only judge steps, or of ranker steps
with `candidate_columns`, needs no definer.

One step of `config/study/scores.json`, shortened:

```json
{
  "name": "judge-qwen38-on-anw",
  "task": "judge",
  "model_name": "Qwen/Qwen3.8-27B-FP8",
  "provider": "vllm-offline",
  "file_directory": "instructions/judge",
  "definition_column": "originele-definitie",
  "post_field_renamer": {"correctheid": "correctheid_anw_judge_qwen38"},
  "client_run_kwargs": {"temperature": 0.7, "max_completion_tokens": 2048}
}
```

The real step renames all six criteria. Step options:

- `definition_column`: the column that a judge scores. It is shown to the
  prompt as `te-beoordelen-definitie`. The value `originele-definitie` judges
  the ANW definition with the same prompt, which gives the ANW baseline.
- `post_field_renamer`: renames the schema fields in the output. Without it,
  judge and feedback steps use `<field>_<step name>`. The definer config
  renames `definitie` to `definitie-v1`.
- Judge scale: each of the six criteria (`begrijpelijkheid`, `betekenis`,
  `gebruik`, `structuur`, `substitueerbaarheid`, `correctheid`) gets a short
  analysis and an integer score from 1 to 5. The prompt anchors the levels: 5
  meets the criterion fully, 4 has a small flaw that does not hinder
  understanding, 3 a clear flaw but is still usable, 2 a serious flaw, 1 does
  not meet the criterion.
- Ranker labels: with two candidates the model answers `1 beter dan 2`,
  `2 beter dan 1` or `evenwaardig` (tie).
- `shuffle-order`: after the last definer or revisor of a pipeline with a
  ranker, a row gets `random-definitie-1`, `random-definitie-2` (and `-3`) and
  `shuffle-order`, the source column of each position, for example
  `["originele-definitie", "definitie-v1"]`. The shuffle is seeded by the row.
- `order_rotation`: the base order is rotated left by this many places before
  it is shown to the ranker. Two ranker steps with rotation 0 and 1 show each
  pair in both orders, which allows a position-bias check. The order that a
  step used is stored in `volgorde_<step name>`.
- `candidate_columns`: a ranker compares the listed columns (for example
  `definitie-gemma4-26b-a4b` against `definitie-qwen36-35b-a3b`) instead of
  the columns in `shuffle-order`. The base order is a shuffle seeded by
  `betekenis_pid` and the column names.
- `revisie-uitgevoerd`: a revisor only runs on rows with feedback. Rows
  without feedback keep `definitie-v1` as the revised definition and get
  `revisie-uitgevoerd=False`.

## Output

```
outputs/study/writer-gemma4-26b-a4b/
  pipeline.json                    config, built steps and run_metadata
  1-definer-<name>/done            dataset after step 1
  1-definer-<name>/attrition.json
  1-definer-<name>/failed_samples.jsonl
  2-judge-<name>/done
  ...
  full_process_done                final dataset
```

- `pipeline.json` has `run_metadata` with the git commit, the versions of
  `def-gen`, `llm-annotator`, `datasets` and `vllm`, and a UTC timestamp.
- `attrition.json` counts rows in, valid rows, invalid rows and the finish
  reasons of the invalid rows. `failed_samples.jsonl` holds the invalid rows.
  Rows whose output does not fit the schema are dropped after the step.
- Load the result with `datasets.load_from_disk("<run>/full_process_done")`.

A row of a writer run has, among others, these columns:

| Column | Example |
| --- | --- |
| `betekenis_pid` | item id |
| `lemmavorm` | `"wasmachine"` |
| `voorbeelden` | `"- De wasmachine staat in de kelder.\n- ..."` |
| `originele-definitie` | the ANW definition |
| `definitie-v1` | the written definition |
| `score_correctheid_judge_qwen38` | `4` |
| `ranking_ranker_qwen38_r0` | `"2 beter dan 1"` |
| `volgorde_ranker-qwen38-r0` | `["definitie-v1", "originele-definitie"]` |
| `shuffle-order` | `["originele-definitie", "definitie-v1"]` |

Columns that start with the step number and name (for example
`2-judge-qwen38_valid_fields`) are bookkeeping columns of the annotator.

## Running the study

Design: pairwise comparisons use 1,000 items. Per-criterion 1-5 scores use
the first 300 of those items. The `eval` split is in random order, so the
first rows are a random subset. The `config/study/` files read it, set
`num_samples` accordingly and set `hub_id` to null, so nothing is pushed. The
models run through offline vLLM and need a GPU.

```bash
# 1. Writers (1,000 items; each run has a definer and ranker steps against ANW)
uv run python scripts/generate_definitions.py config/study/writer-gemma4-26b-a4b.json
uv run python scripts/generate_definitions.py config/study/writer-qwen36-35b-a3b.json

# 2. Merge the definitions of the writers
uv run python scripts/merge_writer_outputs.py \
    outputs/study/writer-gemma4-26b-a4b outputs/study/writer-qwen36-35b-a3b \
    --output_dir outputs/study/head-to-head-input

# 3. Writers against each other (1,000 items) and scores (300 items).
#    The two runs are independent.
uv run python scripts/generate_definitions.py config/study/head-to-head.json
uv run python scripts/generate_definitions.py config/study/scores.json

# 4. Analysis
uv run python scripts/analyze.py \
    outputs/study/writer-gemma4-26b-a4b outputs/study/writer-qwen36-35b-a3b \
    outputs/study/head-to-head outputs/study/scores \
    --output_dir outputs/study/analysis
```

The analysis keeps two item sets. The pair analyses use the items that every
ranker group has (1,000 items). The score analyses use the items that every
score group has (up to 300 items). `coverage.csv` lists the items kept and
dropped per group and table. `self_preference` uses both sets: its win rate
comes from the pair items and its score difference from the scored items.

The scores run has three judge steps per judge model, one for each of the
three systems: the definitions of both writers and the ANW definition
(`definition_column` is `definitie-<writer>` or `originele-definitie`). Step
names and output fields name both judge and judged system, for example the
step `judge-qwen38-on-anw` and the field `begrijpelijkheid_anw_judge_qwen38`.
The analysis maps `definitie-<writer>` to that writer and
`originele-definitie` to "anw".

Primary outcome: per writer, the win rate against ANW. Each item is ranked in
two presentation orders. It counts as a win or a loss only if both orders
agree, and as a tie otherwise (as implemented in `def_gen.analysis`). The win
rate of each single order and the inconsistency rate (items where the orders
disagree) are reported next to it.

`merge_writer_outputs.py` keeps the items that all writers have and adds one
`definitie-<writer>` column per writer. `analyze.py` writes `scores.csv` and
`pairs.csv` (long tables), one CSV per analysis (`coverage`, `attrition`,
`win_rates`, `position_bias`, `order_consistency`, `bradley_terry`,
`writer_mcnemar`, `score_differences`, `score_agreement`, `self_preference`,
`length_bias`, `overlap_items`, `overlap_summary`) and `report.md`, which
renders them.

Resuming: a step whose `done` directory exists and loads is skipped. After an
interruption, run the same command again. The final rows equal those of an
uninterrupted run. `"overwrite": true` in the config deletes the step
directories and starts over.

## Paid APIs

Steps with the provider `openai` or `anthropic` are refused before any client
is created unless the config sets `"allow_paid_api": true`.
`config/study/writer-openai.json` is prepared for a GPT writer (1,000 items) and is refused
as committed (`"allow_paid_api": false`). Install the matching extra
(`uv sync --extra openai`) before enabling it.

## Configs

Study configs in `config/study/`:

- `writer-gemma4-26b-a4b.json` and `writer-qwen36-35b-a3b.json`: one writer
  each, with 1,000 items and ranker steps against ANW (two presentation
  orders per judge).
- `head-to-head.json`: rankers on the definitions of the two writers.
- `scores.json`: three judges with 1-5 scores for each of the three systems on
  300 items of the merged dataset.
- `writer-openai.json`: prepared, refused as committed.

Older examples in `config/`:

- `offline_vllm.example.json`: definer, two feedback steps, revisor, judge and
  a ranker with the `ranker-after-revision` prompt. It sets `hub_id` to the
  author's namespace, so set it to null before running.
- `qwen-36-27b-fp8.json`: a definer and a judge on the 5-example item set.
- `openai_anthropic.example.json`: an OpenAI definer and an Anthropic judge on
  5 items. It runs only with `allow_paid_api` set to true.

Run any config with:

```bash
uv run python scripts/generate_definitions.py config/offline_vllm.example.json
```

## Human evaluation

```bash
uv run python scripts/prepare_for_lexit.py \
    outputs/study/writer-gemma4-26b-a4b/full_process_done \
    lexit.xlsx --test_split_size 100
```

The sheet holds the lemma, the examples, the shuffled definitions
(`random-definitie-1`, `-2`) and `shuffle-order`. The annotators see only the
positions. `shuffle-order` maps a chosen position back to its source. With
`["originele-definitie", "definitie-v1"]`, position 1 is the ANW definition.
`scripts/rank_ira.py` computes agreement between the ranker columns of a
dataset.

## Limitations

- Population: the items are monosemous nouns whose ANW definition has more than
  5 words and that have enough examples. This is the easy end of the
  dictionary. The results do not describe polysemous or rarer entries.
- Contamination: the ANW text can be in the pretraining data of the writers and
  judges. This cannot be controlled. The analysis reports the overlap between
  written and ANW definitions (chrF, longest common substring) as a
  description.
- Judge bias: LLM judges can prefer the first position, longer definitions or
  their own output. The analysis measures position bias, length bias and
  self-preference. It does not remove them.

## Development

| Command | What it does |
| --- | --- |
| `make test` | Run tests and doctests with coverage |
| `make quality` | Lint and format checks (no changes) |
| `make style` | Auto-fix lint issues and reformat |

## License

Apache-2.0. See [LICENSE](LICENSE).
