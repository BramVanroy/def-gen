# def-gen

[![CI](https://github.com/BramVanroy/def-gen/actions/workflows/ci.yml/badge.svg)](https://github.com/BramVanroy/def-gen/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/BramVanroy/def-gen/branch/main/graph/badge.svg)](https://codecov.io/gh/BramVanroy/def-gen)
[![License](https://img.shields.io/github/license/BramVanroy/def-gen)](LICENSE)

Generating and evaluating dictionary definitions with LLMs.

## Installation

```bash
git clone https://github.com/BramVanroy/def-gen.git
cd def-gen
uv sync --all-extras
uv run pre-commit install
```

The extras select the LLM provider: `vllm`, `openai` or `anthropic`. For
example, `uv sync --extra openai` installs only what the OpenAI client needs.

## Usage

A run is described by a JSON config file. The files in `config/` are examples.

```bash
uv run python scripts/generate_definitions.py config/offline_vllm.example.json
```

## Development

| Command | What it does |
| --- | --- |
| `make test` | Run tests and doctests with coverage |
| `make quality` | Lint and format checks (no changes) |
| `make style` | Auto-fix lint issues and reformat |

## License

Apache-2.0. See [LICENSE](LICENSE).
