.PHONY: quality style test

# Lint and format checks (no changes made).
quality:
	uv run ruff check .
	uv run ruff format --check .

# Auto-fix lint violations and reformat.
style:
	uv run ruff check --fix .
	uv run ruff format .

# Run tests and doctests with coverage (also used by CI).
test:
	uv run pytest --cov --cov-report=term-missing --cov-report=xml
