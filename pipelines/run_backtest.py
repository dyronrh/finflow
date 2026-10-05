"""Run a reproducible backtest and write its artefacts.

python pipelines/run_backtest.py --start 2021-01-01 --end 2026-10-05 --out reports/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from backtesting.engine import run_backtest
from data_platform.synthetic import generate_synthetic_market
from quant_core.config import default_strategy_config, load_strategy_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-10-05")
    parser.add_argument("--config", type=Path, help="strategy YAML (default: v0.1.0)")
    parser.add_argument("--seed", type=int, default=42, help="synthetic market seed")
    parser.add_argument("--nav", type=float, default=1_000_000.0)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()

    config = load_strategy_config(args.config) if args.config else default_strategy_config()
    market = generate_synthetic_market(start="2019-01-01", end=args.end, seed=args.seed)
    result = run_backtest(market, config, args.start, args.end, initial_nav=args.nav)

    run_dir = args.out / f"{config.strategy_version}_{result.metadata['config_hash']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(json.dumps(result.to_report(), indent=2, default=str))
    result.equity.to_frame().join(result.benchmark_equity).to_csv(run_dir / "equity.csv")
    result.trades.to_csv(run_dir / "trades.csv", index=False)
    result.holdings.to_csv(run_dir / "holdings.csv", index=False)
    result.rebalances.to_csv(run_dir / "rebalances.csv", index=False)

    print(json.dumps(result.summary, indent=2))
    print(f"\nArtefacts written to {run_dir}")
    print("Synthetic data: results say nothing about real-world performance.")


if __name__ == "__main__":
    main()
