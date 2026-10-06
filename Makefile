.PHONY: install install-real test lint format typecheck api backtest fetch-data backtest-real tune-real tune-real-weights

install:
	uv sync

install-real:
	uv sync --extra real-data

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
	uv run python pipelines/run_backtest.py --source synthetic

fetch-data:
	uv run --extra real-data python pipelines/fetch_real_data.py

backtest-real:
	uv run --extra real-data python pipelines/run_backtest.py --source real

tune-real:
	uv run --extra real-data python pipelines/tune_strategy.py --source real

tune-real-weights:
	uv run --extra real-data python pipelines/tune_strategy.py --source real --grid configs/tuning/factor_weights.yaml
