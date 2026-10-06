"""Parameter tuning without fooling ourselves (README §8.3, §12.2, §12.4).

Protocol
--------
1. **Holdout.** Everything from ``holdout_start`` on is never used to choose
   parameters. It is looked at once, at the end.
2. **Grid.** Every configuration is backtested once over the full window. The
   engine is causal (decisions at ``t`` only use data up to ``t``), so the
   returns of any sub-window depend only on information available by then.
3. **Walk-forward selection** (pre-holdout only): for each test window, the
   configuration with the best objective on the preceding training window is
   selected, and its returns in the test window are stitched together. This is
   the out-of-sample estimate of *the tuning procedure itself*.
4. **Final choice**: the best configuration on the whole pre-holdout period is
   evaluated on the holdout against the base strategy and the benchmark.
5. **Diagnostics**: information coefficients of each factor family, so a weak
   result can be traced to "factors do not predict" vs "portfolio does not
   exploit the signal".

Limitations: stitched walk-forward returns ignore the one-off cost of switching
configuration at a fold boundary; with N configurations tried, the best
in-sample result is optimistic by construction (compare it with the
walk-forward and holdout numbers, not on its own).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtesting.engine import BacktestResult, rebalance_dates, run_backtest
from backtesting.metrics import TRADING_DAYS, performance_summary
from data_platform.features import build_feature_snapshot
from data_platform.market import MarketData
from quant_core.config import FACTOR_FAMILIES, StrategyConfig
from quant_core.factors.definitions import FACTOR_DEFINITIONS
from quant_core.factors.normalization import winsorized_sector_percentile
from quant_core.signals.rules import generate_signals, label_signals, score_universe, scoring_key

DEFAULT_GRID: dict[str, list[object]] = {
    "portfolio.max_turnover_per_rebalance": [0.25, 0.5, 1.0],
    "signals.long_composite_percentile": [0.80, 0.85, 0.90],
    "portfolio.weighting": ["equal_weight", "inverse_volatility"],
}

Objective = Callable[[pd.Series, pd.Series], float]


def information_ratio(returns: pd.Series, benchmark: pd.Series) -> float:
    active = (returns - benchmark).dropna()
    sd = active.std(ddof=1)
    return float(active.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def sharpe(returns: pd.Series, _benchmark: pd.Series) -> float:
    r = returns.dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


OBJECTIVES: dict[str, Objective] = {"information_ratio": information_ratio, "sharpe": sharpe}


# --------------------------------------------------------------------------- grid
def apply_overrides(
    base: StrategyConfig, overrides: dict[str, object], version: str | None = None
) -> StrategyConfig:
    """Return ``base`` with dotted-path overrides.

    Example: ``{"portfolio.weighting": "equal_weight"}``.
    """
    data = base.model_dump()
    for path, value in overrides.items():
        node = data
        *parents, leaf = path.split(".")
        for key in parents:
            node = node[key]
        if leaf not in node:
            raise KeyError(f"unknown config field: {path}")
        node[leaf] = value
    if version:
        data["strategy_version"] = version
    return StrategyConfig.model_validate(data)


def expand_grid(grid: dict[str, list[object]]) -> list[dict[str, object]]:
    keys = list(grid)
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*grid.values())]


class SnapshotCache:
    """Feature snapshots are config-independent and scores only depend on a few
    config fields: compute each (date, scoring config) once across the grid."""

    def __init__(self, market: MarketData) -> None:
        self.market = market
        self._cache: dict[pd.Timestamp, pd.DataFrame] = {}
        self._scores: dict[tuple[pd.Timestamp, str], pd.DataFrame] = {}

    def signals(self, as_of: pd.Timestamp, config: StrategyConfig) -> pd.DataFrame:
        key = (pd.Timestamp(as_of), scoring_key(config))
        if key not in self._scores:
            self._scores[key] = score_universe(self(as_of), config)
        return label_signals(self._scores[key], config)

    def __call__(self, as_of: pd.Timestamp) -> pd.DataFrame:
        key = pd.Timestamp(as_of)
        if key not in self._cache:
            self._cache[key] = build_feature_snapshot(self.market, key)
        return self._cache[key]


# --------------------------------------------------------------------------- walk-forward
def walk_forward_selection(
    returns: pd.DataFrame,
    benchmark: pd.Series,
    train_days: int,
    test_days: int,
    objective: Objective,
) -> tuple[pd.Series, pd.DataFrame]:
    """Stitched out-of-sample returns of the "pick the best on the past" procedure.

    ``returns``: daily returns, one column per configuration.
    """
    dates = returns.index
    stitched, folds = [], []
    test_start = train_days
    while test_start < len(dates):
        train = slice(test_start - train_days, test_start)
        test = slice(test_start, min(test_start + test_days, len(dates)))
        scores = {
            c: objective(returns[c].iloc[train], benchmark.iloc[train]) for c in returns.columns
        }
        valid = {c: s for c, s in scores.items() if np.isfinite(s)}
        chosen = max(valid, key=valid.get) if valid else returns.columns[0]
        stitched.append(returns[chosen].iloc[test])
        folds.append(
            {
                "train_start": dates[train.start],
                "train_end": dates[train.stop - 1],
                "test_start": dates[test.start],
                "test_end": dates[test.stop - 1],
                "chosen_config": chosen,
                "train_score": valid.get(chosen, float("nan")),
            }
        )
        test_start += test_days
    if not stitched:
        raise ValueError(
            f"not enough history for walk-forward: {len(dates)} days < train window {train_days}"
        )
    return pd.concat(stitched), pd.DataFrame(folds)


# --------------------------------------------------------------------------- diagnostics
def factor_information_coefficients(
    market: MarketData,
    config: StrategyConfig,
    provider: Callable[[pd.Timestamp], pd.DataFrame],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Spearman IC between scores at each rebalance and next-period returns.

    Forward return = open of the day after this rebalance → open of the day
    after the next rebalance, i.e. exactly the holding period the backtest uses.
    """
    dates = market.dates[
        (market.dates >= pd.Timestamp(start)) & (market.dates <= pd.Timestamp(end))
    ]
    schedule = list(rebalance_dates(dates, config.portfolio.rebalance_frequency))
    position = {d: i for i, d in enumerate(market.dates)}
    rows = []
    for this, nxt in itertools.pairwise(schedule):
        i0, i1 = position[this] + 1, position[nxt] + 1
        if i1 >= len(market.dates):
            break
        signals = (
            provider.signals(this, config)
            if isinstance(provider, SnapshotCache)
            else generate_signals(provider(this), config)
        )
        if len(signals) < 20:
            continue
        p0 = market.open.iloc[i0].reindex(signals["security_id"])
        p1 = market.open.iloc[i1].reindex(signals["security_id"])
        fwd = pd.Series((p1 / p0 - 1.0).to_numpy(), index=signals.index)
        row: dict[str, object] = {"date": this, "n": int(fwd.notna().sum())}
        for name in ("composite", *FACTOR_FAMILIES):
            score = signals[f"{name}_score"]
            ok = score.notna() & fwd.notna()
            row[f"ic_{name}"] = (
                float(score[ok].rank().corr(fwd[ok].rank())) if ok.sum() >= 20 else np.nan
            )
        # Individual features, ranked within sector exactly as the scorer does.
        for features in FACTOR_DEFINITIONS.values():
            for feature, higher_is_better in features.items():
                if feature not in signals.columns or signals[feature].notna().sum() < 20:
                    continue
                pct = winsorized_sector_percentile(
                    signals, feature, higher_is_better=higher_is_better
                )
                ok = pct.notna() & fwd.notna()
                row[f"ic_feature_{feature}"] = (
                    float(pct[ok].rank().corr(fwd[ok].rank())) if ok.sum() >= 20 else np.nan
                )
        quintile = pd.qcut(signals["composite_score"].rank(method="first"), 5, labels=False)
        by_q = fwd.groupby(quintile).mean()
        row["q5_minus_q1"] = float(by_q.get(4, np.nan) - by_q.get(0, np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def ic_by_year(ic: pd.DataFrame) -> pd.DataFrame:
    """Mean IC of the composite and each family per calendar year."""
    if ic.empty:
        return pd.DataFrame()
    columns = [f"ic_{n}" for n in ("composite", *FACTOR_FAMILIES) if f"ic_{n}" in ic.columns]
    table = ic.groupby(pd.to_datetime(ic["date"]).dt.year)[columns].mean()
    return table.rename(columns=lambda c: c.removeprefix("ic_")).dropna(axis=1, how="all")


def summarize_ic(ic: pd.DataFrame, periods_per_year: int = 12) -> pd.DataFrame:
    out = []
    for column in [c for c in ic.columns if c.startswith("ic_")]:
        s = ic[column].dropna()
        if s.empty:
            continue
        out.append(
            {
                "factor": column.removeprefix("ic_"),
                "mean_ic": s.mean(),
                "t_stat": s.mean() / s.std(ddof=1) * np.sqrt(len(s)) if len(s) > 1 else np.nan,
                "pct_positive": (s > 0).mean(),
                "periods": len(s),
            }
        )
    spread = ic["q5_minus_q1"].dropna() if "q5_minus_q1" in ic else pd.Series(dtype=float)
    if len(spread):
        out.append(
            {
                "factor": "q5_minus_q1_annualized",
                "mean_ic": spread.mean() * periods_per_year,
                "t_stat": spread.mean() / spread.std(ddof=1) * np.sqrt(len(spread)),
                "pct_positive": (spread > 0).mean(),
                "periods": len(spread),
            }
        )
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- driver
@dataclass
class TuningReport:
    grid: pd.DataFrame
    folds: pd.DataFrame
    walk_forward: dict[str, dict[str, float]]
    holdout: dict[str, dict[str, float]]
    ic: pd.DataFrame
    ic_summary: pd.DataFrame
    ic_yearly: pd.DataFrame
    best_config_id: str
    best_overrides: dict[str, object]
    candidate: StrategyConfig
    metadata: dict[str, object] = field(default_factory=dict)
    results: dict[str, BacktestResult] = field(default_factory=dict, repr=False)
    oos_returns: pd.Series = field(default_factory=pd.Series, repr=False)


def _stats(returns: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    s = performance_summary(
        pd.concat([pd.Series([1.0], index=[returns.index[0] - pd.Timedelta(days=1)]), equity])
    )
    s["information_ratio"] = information_ratio(returns, benchmark)
    s["active_return_annualized"] = float((returns - benchmark).mean() * TRADING_DAYS)
    return s


def tune_strategy(
    market: MarketData,
    base: StrategyConfig,
    start: str,
    end: str,
    holdout_start: str,
    grid: dict[str, list[object]] | None = None,
    train_years: float = 5.0,
    test_months: int = 3,
    objective: str = "information_ratio",
    progress: Callable[[str], None] | None = None,
) -> TuningReport:
    grid = grid or DEFAULT_GRID
    score = OBJECTIVES[objective]
    say = progress or (lambda _msg: None)
    provider = SnapshotCache(market)

    combos = expand_grid(grid)
    configs = {f"cfg_{i:02d}": apply_overrides(base, o) for i, o in enumerate(combos)}
    configs = {"base": base, **configs}
    overrides = {"base": {}, **{f"cfg_{i:02d}": o for i, o in enumerate(combos)}}

    results: dict[str, BacktestResult] = {}
    for n, (cid, cfg) in enumerate(configs.items(), 1):
        say(f"[{n}/{len(configs)}] backtest {cid} {overrides[cid]}")
        results[cid] = run_backtest(
            market, cfg, start, end, feature_provider=provider, signal_provider=provider.signals
        )

    returns = pd.DataFrame({cid: r.equity.pct_change() for cid, r in results.items()}).iloc[1:]
    bench = results["base"].benchmark_equity.pct_change().reindex(returns.index)
    holdout_ts = pd.Timestamp(holdout_start)
    pre, post = returns.index < holdout_ts, returns.index >= holdout_ts
    if pre.sum() < 2 or post.sum() < 2:
        raise ValueError("holdout_start must split the backtest window in two non-empty parts")

    # Walk-forward over pre-holdout data; "base" competes like any other config.
    train_days = int(train_years * TRADING_DAYS)
    test_days = int(test_months * TRADING_DAYS / 12)
    oos, folds = walk_forward_selection(returns[pre], bench[pre], train_days, test_days, score)
    oos_bench = bench.reindex(oos.index)
    walk_forward = {
        "tuned_walk_forward": _stats(oos, oos_bench),
        "base_same_windows": _stats(returns["base"].reindex(oos.index), oos_bench),
        "benchmark_same_windows": _stats(oos_bench, oos_bench),
    }
    rsp = None
    if market.equal_weight_reference is not None:
        rsp = market.equal_weight_reference.reindex(results["base"].equity.index).ffill()
        rsp = rsp.pct_change().reindex(returns.index)
        walk_forward["rsp_same_windows"] = _stats(rsp.reindex(oos.index), oos_bench)

    # Final choice on all pre-holdout data, judged once on the holdout.
    in_sample = {c: score(returns[c][pre], bench[pre]) for c in returns.columns}
    best = max((c for c in in_sample if np.isfinite(in_sample[c])), key=in_sample.get)
    holdout = {
        "best_in_sample_config": _stats(returns[best][post], bench[post]),
        "base": _stats(returns["base"][post], bench[post]),
        "benchmark": _stats(bench[post], bench[post]),
    }
    if rsp is not None:
        holdout["rsp"] = _stats(rsp[post], bench[post])

    grid_rows = []
    for cid, r in results.items():
        grid_rows.append(
            {
                "config_id": cid,
                **{k: v for k, v in overrides[cid].items()},
                f"pre_holdout_{objective}": in_sample[cid],
                "pre_holdout_cagr": _stats(returns[cid][pre], bench[pre])["cagr"],
                "pre_holdout_max_drawdown": _stats(returns[cid][pre], bench[pre])["max_drawdown"],
                "avg_turnover": r.summary.get("avg_turnover_per_rebalance"),
                "avg_holdings": r.summary.get("avg_holdings"),
                "times_chosen_walk_forward": int((folds["chosen_config"] == cid).sum()),
            }
        )
    grid_df = pd.DataFrame(grid_rows).sort_values(f"pre_holdout_{objective}", ascending=False)

    say("computing factor information coefficients")
    ic = factor_information_coefficients(market, base, provider, start, holdout_ts)
    candidate = apply_overrides(base, overrides[best], version=f"{base.strategy_version}-tuned-rc")

    return TuningReport(
        grid=grid_df.reset_index(drop=True),
        folds=folds,
        walk_forward=walk_forward,
        holdout=holdout,
        ic=ic,
        ic_summary=summarize_ic(ic),
        ic_yearly=ic_by_year(ic),
        best_config_id=best,
        best_overrides=overrides[best],
        candidate=candidate,
        metadata={
            "objective": objective,
            "n_configs": len(configs),
            "grid": grid,
            "start": start,
            "end": end,
            "holdout_start": holdout_start,
            "train_years": train_years,
            "test_months": test_months,
            "data_version": market.data_version,
            "data_source": market.metadata.get("source", "synthetic"),
            "base_strategy_version": base.strategy_version,
        },
        results=results,
        oos_returns=oos,
    )
