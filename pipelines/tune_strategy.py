"""Tune strategy parameters with walk-forward selection and an untouched holdout.

    uv run python pipelines/tune_strategy.py --source real \\
        --start 2012-01-01 --holdout-start 2023-10-01

Outputs (reports/tuning_<source>_<timestamp>/):
  report.json          walk-forward, holdout and IC summaries
  grid.csv             every configuration tried
  folds.csv            which configuration each walk-forward fold selected
  ic.csv               per-rebalance information coefficients
  candidate.yaml       best pre-holdout configuration (a *candidate*: adopting it
                       as a new strategy version still needs human review, §8.3)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

from _common import add_source_args, load_source
from backtesting.tuning import DEFAULT_GRID, OBJECTIVES, tune_strategy
from quant_core.config import default_strategy_config, load_strategy_config


def _fmt(stats: dict[str, float]) -> str:
    keys = ("cagr", "sharpe_ratio", "information_ratio", "active_return_annualized", "max_drawdown")
    return "  ".join(f"{k}={stats.get(k, float('nan')):+.3f}" for k in keys)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--start", default=None, help="default: 2012-01-01 real, 2016-01-01 synthetic"
    )
    parser.add_argument("--end", default="2026-10-05")
    parser.add_argument("--holdout-start", default=None, help="default: 3 years before --end")
    parser.add_argument("--train-years", type=float, default=5.0)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--objective", choices=sorted(OBJECTIVES), default="information_ratio")
    parser.add_argument(
        "--grid", type=Path, help="YAML mapping of dotted config paths to value lists"
    )
    parser.add_argument("--config", type=Path, help="base strategy YAML (default: v0.1.0)")
    parser.add_argument("--out", type=Path, default=Path("reports"))
    add_source_args(parser)
    args = parser.parse_args()

    start = args.start or ("2012-01-01" if args.source == "real" else "2016-01-01")
    holdout = args.holdout_start or f"{int(args.end[:4]) - 3}{args.end[4:]}"
    base = load_strategy_config(args.config) if args.config else default_strategy_config()
    grid = yaml.safe_load(args.grid.read_text()) if args.grid else DEFAULT_GRID

    market = load_source(args, start, args.end)
    report = tune_strategy(
        market,
        base,
        start,
        args.end,
        holdout,
        grid=grid,
        train_years=args.train_years,
        test_months=args.test_months,
        objective=args.objective,
        progress=print,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out / f"tuning_{args.source}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    report.grid.to_csv(run_dir / "grid.csv", index=False)
    report.folds.to_csv(run_dir / "folds.csv", index=False)
    report.ic.to_csv(run_dir / "ic.csv", index=False)
    (run_dir / "candidate.yaml").write_text(
        "# Candidate produced by pipelines/tune_strategy.py — NOT an approved strategy.\n"
        + yaml.safe_dump(report.candidate.model_dump(mode="json"), sort_keys=False)
    )
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "metadata": report.metadata,
                "best_config_id": report.best_config_id,
                "best_overrides": report.best_overrides,
                "walk_forward": report.walk_forward,
                "holdout": report.holdout,
                "ic_summary": report.ic_summary.to_dict("records"),
            },
            indent=2,
            default=str,
        )
    )

    print("\n=== Factor information coefficients (pre-holdout, base config) ===")
    print(report.ic_summary.round(3).to_string(index=False))
    print(f"\n=== Walk-forward out-of-sample ({len(report.folds)} folds) ===")
    for name, stats in report.walk_forward.items():
        print(f"{name:<26}{_fmt(stats)}")
    print(f"\n=== Holdout from {holdout} (looked at once) ===")
    print(f"best pre-holdout config: {report.best_config_id} {report.best_overrides}")
    for name, stats in report.holdout.items():
        print(f"{name:<26}{_fmt(stats)}")
    print(f"\nArtefacts written to {run_dir}")
    if args.source == "synthetic":
        print("Synthetic data: results say nothing about real-world performance.")


if __name__ == "__main__":
    main()
