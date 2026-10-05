"""Event-driven daily backtester (README §11.1, §12).

Timeline for every rebalance:

* close of day ``t``   – features and signals are computed with data up to ``t``.
* after close ``t``     – a rebalance plan (target weights) is created.
* open of day ``t+1``   – orders fill at the open price; costs are charged.

Signals never fill on the bar that produced them.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from backtesting.metrics import beta, performance_summary
from data_platform.features import build_feature_snapshot
from data_platform.synthetic import SyntheticMarket
from quant_core.config import StrategyConfig
from quant_core.execution.costs import TransactionCostModel
from quant_core.portfolio.construction import build_target_weights
from quant_core.portfolio.rebalance import plan_rebalance
from quant_core.signals.rules import generate_signals

FeatureProvider = Callable[[pd.Timestamp], pd.DataFrame]


@dataclass
class BacktestResult:
    equity: pd.Series
    benchmark_equity: pd.Series
    trades: pd.DataFrame
    holdings: pd.DataFrame
    rebalances: pd.DataFrame
    metadata: dict[str, object]
    summary: dict[str, float] = field(default_factory=dict)

    def to_report(self) -> dict[str, object]:
        return {
            "metadata": self.metadata,
            "summary": self.summary,
            "n_trades": len(self.trades),
            "n_rebalances": len(self.rebalances),
        }


def rebalance_dates(dates: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """Last trading day of each month, or of each half-month for ``biweekly``."""
    s = pd.Series(dates, index=dates)
    if frequency == "monthly":
        keys = [dates.year, dates.month]
    elif frequency == "biweekly":
        keys = [dates.year, dates.month, dates.day > 15]
    else:
        raise ValueError(f"unknown rebalance frequency: {frequency}")
    return pd.DatetimeIndex(s.groupby(keys).max().sort_values().to_numpy())


def run_backtest(
    market: SyntheticMarket,
    config: StrategyConfig,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    initial_nav: float = 1_000_000.0,
    feature_provider: FeatureProvider | None = None,
) -> BacktestResult:
    provider = feature_provider or (lambda as_of: build_feature_snapshot(market, as_of))
    costs = TransactionCostModel.from_config(config.costs)
    pcfg = config.portfolio

    dates = market.dates[
        (market.dates >= pd.Timestamp(start)) & (market.dates <= pd.Timestamp(end))
    ]
    if len(dates) < 2:
        raise ValueError("backtest window needs at least two trading days")
    schedule = set(rebalance_dates(dates, pcfg.rebalance_frequency))

    cash = initial_nav
    shares: dict[str, float] = {}
    pending: tuple[pd.Timestamp, pd.Series] | None = None
    equity, trades, holdings_log, rebalance_log = {}, [], [], []

    for date in dates:
        open_px = market.open.loc[date]
        close_px = market.close.loc[date]

        if pending is not None:
            signal_date, final_weights = pending
            cash, fills = _execute(date, signal_date, final_weights, shares, cash, open_px, costs)
            trades.extend(fills)
            pending = None

        nav = cash + sum(q * close_px[s] for s, q in shares.items())
        equity[date] = nav
        for s, q in shares.items():
            holdings_log.append(
                {"date": date, "security_id": s, "shares": q, "weight": q * close_px[s] / nav}
            )

        if date in schedule and date != dates[-1]:
            current = pd.Series({s: q * close_px[s] / nav for s, q in shares.items()}, dtype=float)
            signals = generate_signals(provider(date), config)
            target = build_target_weights(signals, pcfg, set(current.index))
            plan = plan_rebalance(current, target, pcfg.trade_band, pcfg.max_turnover_per_rebalance)
            pending = (date, plan.final_weights)
            rebalance_log.append(
                {
                    "signal_date": date,
                    "n_targets": len(target),
                    "planned_turnover": plan.turnover,
                    "turnover_capped": plan.turnover_capped,
                }
            )

    equity_s = pd.Series(equity, name="strategy")
    bench_returns = market.close.loc[dates].pct_change().mean(axis=1).fillna(0.0)
    bench = (initial_nav * (1.0 + bench_returns).cumprod()).rename("equal_weight_universe")

    trades_df = pd.DataFrame(trades)
    result = BacktestResult(
        equity=equity_s,
        benchmark_equity=bench,
        trades=trades_df,
        holdings=pd.DataFrame(holdings_log),
        rebalances=pd.DataFrame(rebalance_log),
        metadata=_metadata(market, config, dates, initial_nav),
    )
    summary = performance_summary(equity_s)
    bench_summary = performance_summary(bench)
    summary["beta_vs_equal_weight"] = beta(equity_s.pct_change(), bench.pct_change())
    summary["benchmark_cagr"] = bench_summary.get("cagr", float("nan"))
    summary["benchmark_sharpe"] = bench_summary.get("sharpe_ratio", float("nan"))
    summary["total_costs_usd"] = float(trades_df["cost_usd"].sum()) if len(trades_df) else 0.0
    summary["avg_turnover_per_rebalance"] = (
        float(result.rebalances["planned_turnover"].iloc[1:].mean())
        if len(result.rebalances) > 1
        else float("nan")
    )
    summary["avg_holdings"] = (
        float(result.holdings.groupby("date").size().mean()) if len(result.holdings) else 0.0
    )
    result.summary = summary
    return result


def _execute(
    date: pd.Timestamp,
    signal_date: pd.Timestamp,
    final_weights: pd.Series,
    shares: dict[str, float],
    cash: float,
    open_px: pd.Series,
    costs: TransactionCostModel,
) -> tuple[float, list[dict[str, object]]]:
    """Fill at the open of ``date``; trade notionals are sized net of costs."""
    nav_open = cash + sum(q * open_px[s] for s, q in shares.items())
    names = set(shares) | set(final_weights.index)

    def deltas(investable: float) -> dict[str, float]:
        return {
            s: final_weights.get(s, 0.0) * investable - shares.get(s, 0.0) * open_px[s]
            for s in names
        }

    first_pass = deltas(nav_open)
    estimated_cost = sum(costs.cost(v) for v in first_pass.values())
    notionals = deltas(nav_open - estimated_cost)

    fills = []
    for s in sorted(names):
        notional = notionals[s]
        if abs(notional) < 1e-6:
            continue
        price = float(open_px[s])
        cost = costs.cost(notional)
        shares[s] = shares.get(s, 0.0) + notional / price
        cash -= notional + cost
        if abs(shares[s] * price) < 1e-6:
            del shares[s]
        fills.append(
            {
                "signal_date": signal_date,
                "fill_date": date,
                "security_id": s,
                "side": "BUY" if notional > 0 else "SELL",
                "fill_price": price,
                "notional_usd": notional,
                "cost_usd": cost,
            }
        )
    return cash, fills


def _metadata(
    market: SyntheticMarket,
    config: StrategyConfig,
    dates: pd.DatetimeIndex,
    initial_nav: float,
) -> dict[str, object]:
    config_json = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    return {
        "strategy_version": config.strategy_version,
        "config_hash": hashlib.sha256(config_json.encode()).hexdigest()[:16],
        "data_version": market.data_version,
        "start_date": str(dates[0].date()),
        "end_date": str(dates[-1].date()),
        "initial_nav": initial_nav,
        "execution_assumption": "signal_at_close_t__fill_at_open_t+1",
        "commission_bps": config.costs.commission_bps,
        "half_spread_bps": config.costs.half_spread_bps,
        "slippage_bps": config.costs.slippage_bps,
        "rebalance_frequency": config.portfolio.rebalance_frequency,
        "random_seed": market.seed,
        "git_commit": _git_commit(),
    }


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
