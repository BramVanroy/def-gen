quality:
	ruff check src/def_gen scripts/
	ruff format --check src/def_gen scripts/

style:
	ruff check src/def_gen scripts/ --fix
	ruff format src/def_gen scripts/

setup:
	uv sync --dev
	pre-commit install --hook-type pre-push
