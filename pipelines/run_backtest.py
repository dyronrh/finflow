"""Run a reproducible backtest and write its artefacts.

uv run python pipelines/run_backtest.py --source real --start 2012-01-01
uv run python pipelines/run_backtest.py --source synthetic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import add_source_args, load_source
from backtesting.engine import run_backtest
from quant_core.config import default_strategy_config, load_strategy_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--start", default=None, help="default: 2012-01-01 real, 2021-01-01 synthetic"
    )
    parser.add_argument("--end", default="2026-10-05")
    parser.add_argument("--config", type=Path, help="strategy YAML (default: v0.1.0)")
    parser.add_argument("--nav", type=float, default=1_000_000.0)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    add_source_args(parser)
    args = parser.parse_args()
    start = args.start or ("2012-01-01" if args.source == "real" else "2021-01-01")

    config = load_strategy_config(args.config) if args.config else default_strategy_config()
    market = load_source(args, start, args.end)
    result = run_backtest(market, config, start, args.end, initial_nav=args.nav)

    run_dir = args.out / f"{args.source}_{config.strategy_version}_{result.metadata['config_hash']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(json.dumps(result.to_report(), indent=2, default=str))
    result.equity.to_frame().join(result.benchmark_equity).to_csv(run_dir / "equity.csv")
    result.trades.to_csv(run_dir / "trades.csv", index=False)
    result.holdings.to_csv(run_dir / "holdings.csv", index=False)
    result.rebalances.to_csv(run_dir / "rebalances.csv", index=False)

    print(json.dumps(result.summary, indent=2))
    bias = result.summary.get("estimated_survivorship_bias_cagr")
    if bias is not None:
        print(
            f"\nEstimated residual survivorship bias: {bias:+.2%} per year "
            "(equal-weight universe CAGR minus RSP CAGR; RSP charges ~0.20%/yr)"
        )
    print(f"\nArtefacts written to {run_dir}")
    if args.source == "synthetic":
        print("Synthetic data: results say nothing about real-world performance.")


if __name__ == "__main__":
    main()
