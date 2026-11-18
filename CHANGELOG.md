# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `def_gen.analysis`: long score and pair tables from writer, ANW and head-to-head runs, win rates with item bootstrap and sign test, position bias and order consistency, Bradley-Terry abilities (choix), McNemar between writers, per-criterion score differences with Wilcoxon and Holm, ordinal judge agreement, self-preference and length bias, overlap with ANW (chrF, longest common substring), and human-LLM agreement.
- `scripts/analyze.py` writes all tables as CSV and a `report.md`.
- Dependency `choix`.
- `scripts/merge_writer_outputs.py` joins the definitions of several writer runs on `betekenis_pid`.
- Ranker option `candidate_columns` (2 or 3 columns, no definer needed) and `order_rotation` (a ranker step shows the candidates in a rotated order).
- `definition_column` for judge steps, to judge any definition column, for example the ANW one.
- Pipelines that consist only of judge steps.
- `from_hub_id` accepts a local `save_to_disk` directory.
- `scripts/make_item_split.py` writes a seeded dev/eval split.
- Study configs in `config/study/`: two writers (1,000 items, ranker steps against ANW), head-to-head ranking, `scores.json` (three judges on three systems, 300 items) and a prepared OpenAI writer.
- `allow_paid_api` (default false). Steps with the provider `openai` or `anthropic` are refused without it.
- Per-step `attrition.json` and `failed_samples.jsonl`.
- `pipeline.json` has `run_metadata` (git commit, package versions, UTC timestamp).
- Agreement statistics: percent agreement, Gwet's AC1, ordinal and interval Krippendorff's alpha, per-rater preference rates.
- README sections on data, pipeline, output, the study commands, paid APIs and limitations. Usage docstrings in all scripts.

### Changed

- Judge criteria use an anchored 1-5 scale instead of 0-100.
- One judge prompt for all definition columns (`instructions/judge`).
- `num_samples` also applies to datasets from `from_hub_id`.
- A revisor runs only on rows with feedback, in fresh and resumed runs alike.
- Pipeline validation checks directly that the only definer is the first step.
- Agreement bootstraps skip undefined resamples and report how many.
- `pipeline.json` stores the built steps instead of the unbuilt config steps.
- `scripts/quick_view.py` and `scripts/check_lengths.py` take their input as arguments.
- CI passes the Codecov token to the upload step only.
- `config/openai_anthropic.example.json` sets `allow_paid_api` to false.

### Fixed

- Typos in the ranker and feedback prompts.
- Gwet's AC1 on resampled data with repeated index labels.
- The error for a step that precedes the definer named the wrong cause.

### Removed

- `instructions/judge-after-revision`.
