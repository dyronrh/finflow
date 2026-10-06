.PHONY: install install-real test lint format typecheck api backtest fetch-data backtest-real tune-real tune-real-weights score-real research research-real train-ml train-ml-real models paper-status paper-propose paper-reconcile web-install web-dev web-build app app-real static static-real deploy-vercel

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

research:  ## candidate-factor study on synthetic data
	uv run python pipelines/research_factors.py --source synthetic

research-real:  ## candidate-factor study on real cached data (run make fetch-data first)
	uv run --extra real-data python pipelines/research_factors.py --source real

score-real:
	uv run --extra real-data python pipelines/score_universe.py --source real --tickers $(TICKERS) --top 10

train-ml:
	uv run --extra ml python pipelines/train_ml.py --source synthetic

train-ml-real:
	uv run --extra real-data --extra ml python pipelines/train_ml.py --source real

models:
	uv run python pipelines/model_registry.py list

paper-status:
	uv run --extra real-data python pipelines/paper_trade.py status

paper-propose:
	uv run --extra real-data python pipelines/paper_trade.py propose

paper-reconcile:
	uv run --extra real-data python pipelines/paper_trade.py sync && \
	uv run --extra real-data python pipelines/paper_trade.py reconcile

web-install:
	cd apps/web && npm ci

web-dev:  ## UI with hot reload on :5173 (run `make api` in another terminal)
	cd apps/web && npm run dev

web-build:
	cd apps/web && npm run build

app: web-build  ## UI + API on http://localhost:8000 (synthetic data)
	uv run uvicorn apps.api.main:app --port 8000

app-real: web-build  ## UI + API on real cached data
	FINFLOW_DATA_SOURCE=real uv run --extra real-data uvicorn apps.api.main:app --port 8000

static:  ## static site with synthetic data in apps/web/dist (deployable to Vercel)
	cd apps/web && npm run build:static
	uv run python pipelines/export_static.py --source synthetic

static-real:  ## static site with real cached data (run make fetch-data first)
	cd apps/web && npm run build:static
	uv run --extra real-data python pipelines/export_static.py --source real

deploy-vercel:  ## upload apps/web/dist to Vercel (first time: log in and create the project)
	npx vercel@latest deploy apps/web/dist --prod
