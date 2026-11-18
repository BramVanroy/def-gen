# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `def_gen.analysis`: long score and pair tables from writer, ANW and head-to-head runs, win rates with item bootstrap and sign test, position bias and order consistency, Bradley-Terry abilities (choix), McNemar between writers, per-criterion score differences with Wilcoxon and Holm, ordinal judge agreement, self-preference and length bias, overlap with ANW (chrF, longest common substring), and human-LLM agreement.
- `scripts/analyze.py` writes all tables as CSV and a `report.md`.
- Dependency `choix`.
