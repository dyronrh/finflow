"""Sprint 7: walk-forward ML ranking model vs the explainable rule strategy.

    uv run --extra real-data --extra ml python pipelines/train_ml.py --source real
    uv run --extra ml python pipelines/train_ml.py --source synthetic --start 2018-01-01

Steps
  1. point-in-time panel (features at each rebalance, next-period return target)
  2. walk-forward predictions (purged, rolling window, periodic refits)
  3. IC of ML vs composite, before and inside the holdout
  4. backtests with the same engine: rules, ML, 50/50 blend
  5. final model fitted on pre-holdout data only; permutation importance and
     drift measured on the holdout
  6. registered as PENDING_REVIEW (approve with pipelines/model_registry.py)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from _common import add_source_args, load_source
from backtesting.engine import rebalance_dates, run_backtest
from backtesting.tuning import SnapshotCache, _git_commit_safe, window_stats
from ml.dataset import FEATURE_SCHEMA_VERSION, build_panel, feature_columns
from ml.monitoring import drift_report
from ml.registry import DEFAULT_ROOT, ModelMetadata, ModelRegistry
from ml.strategy import ml_signal_provider
from ml.training import (
    MODEL_TYPES,
    daily_ic,
    fit_final,
    hyperparameters,
    ic_summary,
    permutation_importance_ic,
    walk_forward_predict,
)
from quant_core.config import default_strategy_config, load_strategy_config


def _row(name: str, stats: dict[str, float]) -> str:
    keys = ("cagr", "sharpe_ratio", "information_ratio", "active_return_annualized", "max_drawdown")
    return f"{name:<22}" + "  ".join(f"{k}={stats.get(k, float('nan')):+.3f}" for k in keys)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--start", default=None, help="default: 2012-01-01 real, 2016-01-01 synthetic"
    )
    parser.add_argument("--end", default="2026-10-05")
    parser.add_argument("--holdout-start", default=None, help="default: 3 years before --end")
    parser.add_argument("--model", choices=MODEL_TYPES, default="gbm")
    parser.add_argument("--train-years", type=float, default=5.0)
    parser.add_argument("--retrain-months", type=int, default=3)
    parser.add_argument("--config", type=Path, help="strategy YAML (default: v0.1.0)")
    parser.add_argument("--registry", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("reports"))
    add_source_args(parser)
    args = parser.parse_args()

    start = args.start or ("2012-01-01" if args.source == "real" else "2016-01-01")
    holdout = pd.Timestamp(args.holdout_start or f"{int(args.end[:4]) - 3}{args.end[4:]}")
    config = load_strategy_config(args.config) if args.config else default_strategy_config()
    market = load_source(args, start, args.end)

    dates = market.dates[(market.dates >= start) & (market.dates <= args.end)]
    schedule = list(rebalance_dates(dates, config.portfolio.rebalance_frequency))
    cache = SnapshotCache(market)
    print(f"building panel over {len(schedule)} rebalance dates…")
    panel = build_panel(market, config, cache.signals, schedule)
    print(f"panel: {len(panel):,} rows, {panel['date'].nunique()} dates")

    print(
        f"walk-forward {args.model}: train {args.train_years}y, refit every {args.retrain_months}m"
    )
    wf = walk_forward_predict(panel, args.model, args.train_years, args.retrain_months)
    if wf.predictions.empty:
        raise SystemExit("not enough history to fit a first model; use an earlier --start")
    merged = panel.merge(wf.predictions, on=["date", "security_id"], how="inner")
    pre, post = merged[merged["date"] < holdout], merged[merged["date"] >= holdout]

    ic = {
        "pre_holdout": {
            "ml": ic_summary(daily_ic(pre, "ml_score")),
            "composite": ic_summary(daily_ic(pre, "composite_score")),
        },
        "holdout": {
            "ml": ic_summary(daily_ic(post, "ml_score")),
            "composite": ic_summary(daily_ic(post, "composite_score")),
        },
    }

    # Backtests over the span where the model exists, same engine and costs.
    first = wf.predictions["date"].min()
    results = {
        "rules": run_backtest(market, config, first, args.end, signal_provider=cache.signals),
        "ml": run_backtest(
            market,
            config,
            first,
            args.end,
            signal_provider=ml_signal_provider(cache.signals, wf.predictions, 1.0),
        ),
        "blend_50": run_backtest(
            market,
            config,
            first,
            args.end,
            signal_provider=ml_signal_provider(cache.signals, wf.predictions, 0.5),
        ),
    }
    bench = results["rules"].benchmark_equity.pct_change().iloc[1:]
    backtests: dict[str, dict[str, dict[str, float]]] = {"pre_holdout": {}, "holdout": {}}
    for name, res in results.items():
        r = res.equity.pct_change().iloc[1:]
        backtests["pre_holdout"][name] = window_stats(
            r[r.index < holdout], bench[bench.index < holdout]
        )
        backtests["holdout"][name] = window_stats(
            r[r.index >= holdout], bench[bench.index >= holdout]
        )
    backtests["pre_holdout"]["benchmark"] = window_stats(
        bench[bench.index < holdout], bench[bench.index < holdout]
    )
    backtests["holdout"]["benchmark"] = window_stats(
        bench[bench.index >= holdout], bench[bench.index >= holdout]
    )

    # Final model: fitted only on labels known before the holdout.
    model, train = fit_final(panel, holdout, args.model, args.train_years)
    holdout_rows = panel[(panel["date"] >= holdout) & panel["target"].notna()]
    importance = permutation_importance_ic(model, holdout_rows) if len(holdout_rows) else {}
    recent = panel[panel["date"] >= panel["date"].max() - pd.DateOffset(months=6)]
    features = feature_columns()
    drift = drift_report(
        train,
        recent,
        features,
        pd.Series(model.predict(train[features])),
        pd.Series(model.predict(recent[features])),
        training_ic=ic["pre_holdout"]["ml"]["mean_ic"],
        recent_ic=ic["holdout"]["ml"]["mean_ic"],
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out / f"ml_{args.source}_{args.model}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    wf.predictions.to_csv(run_dir / "predictions.csv", index=False)
    wf.fits.to_csv(run_dir / "fits.csv", index=False)
    report = {
        "model": args.model,
        "holdout_start": str(holdout.date()),
        "ic": ic,
        "backtests": backtests,
        "feature_importance": importance,
        "drift": {
            "statuses": drift.statuses,
            "overall": drift.overall,
            "feature_psi": drift.feature_psi,
            "score_psi": drift.score_psi,
        },
        "data_version": market.data_version,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))

    registered = None
    if not args.no_register:
        registry = ModelRegistry(args.registry)
        name = f"ranker_{args.model}"
        meta = ModelMetadata(
            model_name=name,
            model_version=registry.next_version(name),
            model_type=args.model,
            training_data_version=market.data_version,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            feature_names=features,
            train_start_date=str(train["date"].min().date()),
            train_end_date=str(train["date"].max().date()),
            validation_dates=[
                f"{f.train_start.date()}..{f.fit_date.date()}" for f in wf.fits.itertuples()
            ],
            test_dates=[str(holdout.date()), str(pd.Timestamp(args.end).date())],
            metrics={"ic": ic, "backtests": backtests, "drift_overall": drift.overall},
            feature_importance=importance,
            hyperparameters=hyperparameters(model),
            git_commit=_git_commit_safe(),
        )
        registered = registry.register(model, meta)

    print("\n=== IC (Spearman, per rebalance) ===")
    for window, block in ic.items():
        for name, s in block.items():
            print(
                f"{window:<12} {name:<10} mean_ic={s['mean_ic']:+.4f}  t={s['t_stat']:+.2f}  "
                f"pct_pos={s['pct_positive']:.2f}  n={s['periods']}"
            )
    for window, block in backtests.items():
        print(f"\n=== Backtest {window} (same engine, costs and limits) ===")
        for name, s in block.items():
            print(_row(name, s))
    print("\n=== Permutation importance on holdout (IC drop) ===")
    for k, v in list(importance.items())[:10]:
        print(f"  {k:<26}{v:+.4f}")
    print(f"\n=== Drift: {drift.overall} ===  {drift.statuses}")
    worst = sorted(drift.feature_psi.items(), key=lambda kv: -(kv[1] if kv[1] == kv[1] else -1))[:5]
    print("  highest PSI: " + ", ".join(f"{k}={v:.3f}" for k, v in worst))
    if registered:
        print(
            f"\nregistered {registered} as PENDING_REVIEW "
            "(approve: python pipelines/model_registry.py approve ...)"
        )
    print(f"artefacts: {run_dir}")
    if args.source == "synthetic":
        print("Synthetic data: results say nothing about real-world performance.")


if __name__ == "__main__":
    main()
