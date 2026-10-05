.PHONY: install test lint format typecheck api backtest

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy packages

api:
	uv run uvicorn apps.api.main:app --reload

backtest:
	uv run python pipelines/run_backtest.py
