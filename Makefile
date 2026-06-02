.PHONY: quality style test typecheck serve-docs

PACKAGE = src/def_gen

quality:
	uv run interrogate -vv
	uv run ruff check $(PACKAGE) tests/
	uv run ruff format --check $(PACKAGE) tests/

style:
	uv run ruff check $(PACKAGE) tests/ --fix
	uv run ruff format $(PACKAGE) tests/

test:
	uv run pytest --cov=$(PACKAGE) --cov-report=term-missing --cov-report=xml

typecheck:
	uv run mypy $(PACKAGE) tests/

# Only used for doc-testing, not for remote doc deployment
DOCS_BRANCH ?= tmp-gh-pages
DOCS_VERSION ?= 0.0.0
DOCS_ALIAS ?= latest
DOCS_ADDR ?= 127.0.0.1:8000
DOCS_SOURCE_REF ?= main

serve-docs:
	DOCS_SOURCE_REF=$(DOCS_SOURCE_REF) uv run mike deploy --branch $(DOCS_BRANCH) --update-aliases $(DOCS_VERSION) $(DOCS_ALIAS)
	uv run mike set-default --branch $(DOCS_BRANCH) $(DOCS_ALIAS)
	uv run mike serve -b $(DOCS_BRANCH) -a $(DOCS_ADDR)
