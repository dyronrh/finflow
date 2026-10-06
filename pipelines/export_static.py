"""Export a static snapshot of the dashboard for serverless hosting (e.g. Vercel).

    uv run python pipelines/export_static.py --source synthetic --out apps/web/dist/data
    uv run --extra real-data python pipelines/export_static.py --source real

Every file is produced by calling the real API endpoints in-process, so the
static site shows exactly what the live API would. Run it after
`npm run build:static` (see `make static` / `make static-real`).

Paper-trading data is *not* exported unless --include-paper is given: a
static site is public by default.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from _common import ROOT, add_source_args, load_source

sys.path.insert(0, str(ROOT))

import pandas as pd
from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.services import dashboard as dashboard_module
from apps.api.services.dashboard import DashboardService, get_dashboard_service
from apps.api.services.research import ResearchService, get_research_service
from quant_core.config import default_strategy_config, load_strategy_config

BACKTEST_YEARS = (1, 3, 5, 10)


def write(out: Path, name: str, payload: object) -> int:
    path = out / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    path.write_text(data, encoding="utf-8")
    return len(data)


FIELDS = ("time", "open", "high", "low", "close", "volume")


def compact_candles(body: dict) -> dict:
    """Columnar candles (one array per field): less than half the size of
    row objects. The web client expands them (lib/api.ts)."""
    candles = body["candles"]

    def price(v: float) -> float:
        return round(v, 2 if v >= 1 else 4)

    return {
        "security_id": body["security_id"],
        "format": "columnar",
        "time": [c["time"] for c in candles],
        "open": [price(c["open"]) for c in candles],
        "high": [price(c["high"]) for c in candles],
        "low": [price(c["low"]) for c in candles],
        "close": [price(c["close"]) for c in candles],
        "volume": [int(c["volume"]) for c in candles],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=ROOT / "apps" / "web" / "dist" / "data")
    parser.add_argument("--config", type=Path, help="strategy YAML (default: v0.1.0)")
    parser.add_argument("--price-years", type=int, default=5, help="years of candles per security")
    parser.add_argument("--history-months", type=int, default=24)
    parser.add_argument(
        "--include-paper", action="store_true", help="publish the local paper account"
    )
    parser.add_argument("--max-securities", type=int, help="limit (quick trial)")
    add_source_args(parser)
    args = parser.parse_args()

    started = time.time()
    config = load_strategy_config(args.config) if args.config else default_strategy_config()
    market = load_source(args, "2016-01-01", datetime.now().date().isoformat())
    research = ResearchService(market, config)
    dashboard = DashboardService(research, args.source)
    app.dependency_overrides[get_research_service] = lambda: research
    app.dependency_overrides[get_dashboard_service] = lambda: dashboard
    client = TestClient(app)

    def get(url: str) -> dict | list:
        response = client.get(url)
        response.raise_for_status()
        return response.json()

    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    size = 0

    overview = get("/v1/overview")
    size += write(out, "overview.json", overview)
    size += write(out, "rankings.json", get("/v1/rankings?limit=500"))
    securities = [s for s in get("/v1/securities") if s["has_prices"]]
    if args.max_securities:
        securities = securities[: args.max_securities]
    size += write(out, "securities.json", securities)
    print(f"overview, rankings, {len(securities)} securities listed")

    size += write(out, "portfolio.json", get("/v1/portfolio/model"))
    for years in BACKTEST_YEARS:
        t = time.time()
        size += write(out, f"backtest-{years}.json", get(f"/v1/backtest?years={years}"))
        print(f"backtest {years}y ({time.time() - t:.0f}s)")

    if args.include_paper:
        size += write(out, "paper.json", get("/v1/paper"))
    else:
        dashboard_module.PAPER_DIR = Path("/nonexistent")  # publish the empty state only
        size += write(out, "paper.json", get("/v1/paper"))

    price_start = (
        pd.Timestamp(overview["as_of_date"]) - pd.DateOffset(years=args.price_years)
    ).date()
    for i, s in enumerate(securities, 1):
        sid = s["security_id"]
        base = f"securities/{sid}"
        size += write(out, f"{base}/analysis.json", get(f"/v1/securities/{sid}/analysis"))
        prices = compact_candles(get(f"/v1/securities/{sid}/prices?start={price_start}"))
        size += write(out, f"{base}/prices.json", prices)
        history = get(f"/v1/securities/{sid}/score-history?months={args.history_months}")
        size += write(out, f"{base}/history.json", history)
        if i % 50 == 0 or i == len(securities):
            print(f"securities: {i}/{len(securities)}")

    meta = {
        "generated_at": datetime.now(UTC).isoformat(),
        "data_source": args.source,
        "data_version": market.data_version,
        "strategy_version": config.strategy_version,
        "price_history_years": args.price_years,
        "securities": len(securities),
    }
    size += write(out, "meta.json", meta)
    print(
        f"\nwrote {size / 1e6:.1f} MB of JSON to {out} in {time.time() - started:.0f}s "
        f"({args.source} data, strategy {config.strategy_version})"
    )


if __name__ == "__main__":
    main()
