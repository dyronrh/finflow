"""Candidate-factor evaluation (README §8, §12.4).

Protocol
--------
1. **Universe.** At each month-end the eligible universe of the strategy
   (price, market cap, liquidity, index membership on that date).
2. **Score.** Each candidate is turned into a winsorized sector-relative
   percentile, exactly as the production scorer does, with the sign fixed
   *a priori* (``higher_is_better``): a factor is never flipped after looking
   at the data.
3. **Forward returns.** Open of t+1 to the open after the rebalance ``h``
   months later (the holding period the backtest uses). A security that stops
   trading is valued at its last price, like the backtest liquidation, so
   delisted losers are not dropped.
4. **Statistics** on the research sample only: Spearman IC at 1/3/6 months
   with Newey-West t-statistics (overlapping horizons), quintile spreads,
   top-quintile excess over the equal-weighted universe, monotonicity,
   turnover, coverage, stability across halves and years.
5. **Multiple testing.** With ~30 candidates some will look good by chance:
   Benjamini-Hochberg q-values across candidates, plus the stricter
   |t| > 3 hurdle (Harvey, Liu & Zhu 2016).
6. **Holdout.** Reported separately, never used to select. A factor is
   *accepted* only if it passes in the research sample and, in the holdout,
   keeps a positive IC with a Newey-West t of at least 1 (a few years of
   holdout rarely reach significance on their own, but a sign that only
   survives as noise is not confirmation).
7. **Composite** of the accepted factors (equal weights), judged on the
   holdout.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtesting.engine import rebalance_dates
from data_platform.features import build_feature_snapshot
from data_platform.market import MarketData
from quant_core.config import StrategyConfig
from quant_core.factors.definitions import FACTOR_DEFINITIONS
from quant_core.factors.normalization import winsorized_sector_percentile
from quant_core.signals.rules import apply_eligibility

FeatureProvider = Callable[[pd.Timestamp], pd.DataFrame]
HORIZONS = (1, 3, 6)
MIN_CROSS_SECTION = 30
T_HURDLE = 3.0
HOLDOUT_T = 1.0


@dataclass(frozen=True)
class Candidate:
    name: str
    higher_is_better: bool
    family: str
    hypothesis: str
    column: str = ""

    @property
    def feature(self) -> str:
        return self.column or self.name


CANDIDATES: tuple[Candidate, ...] = (
    # value
    Candidate("earnings_yield", True, "value", "cheap on earnings outperforms"),
    Candidate("fcf_yield", True, "value", "cheap on free cash flow outperforms"),
    Candidate("ev_ebitda", False, "value", "low EV/EBITDA outperforms"),
    Candidate("sales_yield", True, "value", "cheap on sales outperforms"),
    # growth
    Candidate("revenue_growth", True, "growth", "faster sales growth outperforms"),
    Candidate("eps_growth", True, "growth", "faster earnings growth outperforms"),
    Candidate("asset_growth", False, "investment", "aggressive asset growth underperforms"),
    # profitability / quality
    Candidate("roic", True, "quality", "high return on capital outperforms"),
    Candidate("gross_margin", True, "quality", "high gross margin outperforms"),
    Candidate("fcf_margin", True, "quality", "high FCF margin outperforms"),
    Candidate("net_debt_ebitda", False, "quality", "low leverage outperforms"),
    Candidate("gross_profitability", True, "quality", "gross profit / capital (Novy-Marx)"),
    Candidate("accruals", False, "quality", "earnings not backed by cash underperform"),
    Candidate("gross_margin_volatility", False, "quality", "stable margins outperform"),
    Candidate("earnings_volatility", False, "quality", "stable earnings outperform"),
    # momentum / reversal
    Candidate("mom_12_1", True, "momentum", "12-1 month winners keep winning"),
    Candidate("mom_6m", True, "momentum", "6 month winners keep winning"),
    Candidate("mom_3m", True, "momentum", "3 month winners keep winning"),
    Candidate("high_52w_ratio", True, "momentum", "stocks near their 52w high outperform"),
    Candidate("ret_1m", False, "reversal", "last month's losers rebound"),
    # low risk
    Candidate("volatility_63d", False, "low_risk", "low recent volatility outperforms"),
    Candidate("volatility_252d", False, "low_risk", "low volatility outperforms"),
    Candidate("beta_252d", False, "low_risk", "low beta outperforms (risk-adjusted)"),
    # size / issuance
    Candidate("log_market_cap", False, "size", "smaller companies outperform"),
    Candidate("net_issuance_1y", False, "issuance", "buybacks beat dilution"),
    # revisions
    Candidate("eps_rev_30d", True, "revisions", "upward EPS revisions outperform"),
    Candidate("eps_rev_90d", True, "revisions", "upward EPS revisions outperform"),
    Candidate("revisions_up_ratio", True, "revisions", "more analysts revising up"),
)

IN_STRATEGY = {f for features in FACTOR_DEFINITIONS.values() for f in features}


# --------------------------------------------------------------------------- statistics
def newey_west_t(values: pd.Series | np.ndarray, lags: int) -> float:
    """t-statistic of the mean with a Bartlett-kernel (Newey-West) variance."""
    x = np.asarray(pd.Series(values).dropna(), dtype=float)
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    var = e @ e / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1)
        var += 2.0 * weight * (e[lag:] @ e[:-lag]) / n
    if var <= 0:
        return float("nan")
    return float(x.mean() / math.sqrt(var / n))


def two_sided_p(t: float) -> float:
    return float(math.erfc(abs(t) / math.sqrt(2.0))) if np.isfinite(t) else float("nan")


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """BH-adjusted q-values; NaN p-values stay NaN and do not count as tests."""
    p = p_values.dropna().sort_values()
    m = len(p)
    if m == 0:
        return pd.Series(np.nan, index=p_values.index)
    ranked = p * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked.to_numpy()[::-1])[::-1].clip(max=1.0)
    return pd.Series(q, index=p.index).reindex(p_values.index)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    if ok.sum() < MIN_CROSS_SECTION:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(a[ok].rank().corr(b[ok].rank()))


# --------------------------------------------------------------------------- panel
@dataclass
class FactorPanel:
    """Per-date observations; everything else is derived from it."""

    ic: pd.DataFrame  # (date, factor) x horizon
    quintiles: pd.DataFrame  # (date, factor) x Q1..Q5 excess 1m return over universe mean
    top_turnover: pd.DataFrame  # date x factor
    coverage: pd.DataFrame  # date x factor
    correlations: list[pd.DataFrame] = field(default_factory=list)
    dates: list[pd.Timestamp] = field(default_factory=list)


def _forward_returns(
    market: MarketData, schedule: Sequence[pd.Timestamp], horizons: Sequence[int]
) -> dict[tuple[pd.Timestamp, int], pd.Series]:
    position = {d: i for i, d in enumerate(market.dates)}
    # Last known price after a security stops trading = liquidation value.
    opens = market.open.ffill()
    last_trade = market.open.apply(pd.Series.last_valid_index)
    out = {}
    for k, date in enumerate(schedule):
        i0 = position[date] + 1
        if i0 >= len(market.dates):
            continue
        p0 = market.open.iloc[i0]  # must trade at entry
        for h in horizons:
            if k + h >= len(schedule):
                continue
            i1 = position[schedule[k + h]] + 1
            if i1 >= len(market.dates):
                continue
            p1 = opens.iloc[i1]
            # Not traded at all after entry -> unknown, not a 0% return.
            traded_after = last_trade.reindex(p0.index) >= market.dates[i0]
            out[(date, h)] = (p1 / p0 - 1.0).where(traded_after)
    return out


def build_panel(
    market: MarketData,
    config: StrategyConfig,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    candidates: Sequence[Candidate] = CANDIDATES,
    provider: FeatureProvider | None = None,
    horizons: Sequence[int] = HORIZONS,
    progress: Callable[[str], None] | None = None,
) -> FactorPanel:
    provider = provider or (lambda d: build_feature_snapshot(market, d))
    say = progress or (lambda _m: None)
    window = market.dates[
        (market.dates >= pd.Timestamp(start)) & (market.dates <= pd.Timestamp(end))
    ]
    schedule = list(rebalance_dates(window, "monthly"))
    # Forward windows may run past ``end`` (they are realised later, not used to decide).
    full = list(rebalance_dates(market.dates, "monthly"))
    tail = [d for d in full if d > schedule[-1]] if schedule else []
    fwd = _forward_returns(market, schedule + tail[: max(horizons)], horizons)

    ic_rows, q_rows, turn_rows, cov_rows, corrs, used = [], [], [], [], [], []
    previous_top: dict[str, set[str]] = {}
    for n, date in enumerate(schedule, 1):
        if (date, horizons[0]) not in fwd:
            continue
        snap = apply_eligibility(provider(date), config)
        universe = snap[snap["eligible"]].set_index("security_id")
        if len(universe) < MIN_CROSS_SECTION:
            continue
        frame = universe.reset_index()
        pcts = {}
        for c in candidates:
            if c.feature not in frame.columns or frame[c.feature].notna().sum() < MIN_CROSS_SECTION:
                continue
            pct = winsorized_sector_percentile(
                frame, c.feature, higher_is_better=c.higher_is_better
            )
            if pct.nunique() < 5:  # constant feature: nothing to rank
                continue
            pcts[c.name] = pd.Series(pct.to_numpy(), index=universe.index)
        if not pcts:
            continue
        used.append(date)
        r1 = fwd[(date, horizons[0])].reindex(universe.index)
        cov_row: dict[str, object] = {"date": date}
        turn_row: dict[str, object] = {"date": date}
        for name, pct in pcts.items():
            row: dict[str, object] = {"date": date, "factor": name}
            for h in horizons:
                r = fwd.get((date, h))
                row[f"ic_{h}m"] = (
                    _spearman(pct, r.reindex(universe.index)) if r is not None else np.nan
                )
            ic_rows.append(row)

            ok = pct.notna() & r1.notna()
            cov_row[name] = float(pct.notna().mean())
            if ok.sum() >= 5 * 10:
                bucket = pd.qcut(pct[ok].rank(method="first"), 5, labels=False)
                excess = r1[ok] - r1[ok].mean()
                means = excess.groupby(bucket).mean()
                q_rows.append(
                    {"date": date, "factor": name, **{f"Q{q + 1}": means.get(q) for q in range(5)}}
                )
                top = set(pct[ok][bucket == 4].index)
                if previous_top.get(name):
                    turn_row[name] = 1.0 - len(top & previous_top[name]) / len(previous_top[name])
                previous_top[name] = top
        cov_rows.append(cov_row)
        turn_rows.append(turn_row)
        with np.errstate(invalid="ignore", divide="ignore"):
            corrs.append(pd.DataFrame(pcts).corr())
        if n % 12 == 0:
            say(f"factor panel: {date.date()} ({n}/{len(schedule)})")

    def frame_of(rows, index):
        df = pd.DataFrame(rows)
        return df.set_index(index) if not df.empty else df

    return FactorPanel(
        ic=frame_of(ic_rows, ["date", "factor"]),
        quintiles=frame_of(q_rows, ["date", "factor"]),
        top_turnover=frame_of(turn_rows, "date"),
        coverage=frame_of(cov_rows, "date"),
        correlations=corrs,
        dates=used,
    )


# --------------------------------------------------------------------------- summaries
def _slice(panel: FactorPanel, start: pd.Timestamp | None, end: pd.Timestamp | None):
    def keep(dates: pd.Index) -> np.ndarray:
        mask = np.ones(len(dates), dtype=bool)
        if start is not None:
            mask &= dates >= start
        if end is not None:
            mask &= dates < end
        return mask

    def by_level(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        dates = (
            df.index.get_level_values("date") if isinstance(df.index, pd.MultiIndex) else df.index
        )
        return df[keep(pd.DatetimeIndex(dates))]

    corrs = [
        c
        for d, c in zip(panel.dates, panel.correlations, strict=True)
        if keep(pd.DatetimeIndex([d]))[0]
    ]
    return (
        by_level(panel.ic),
        by_level(panel.quintiles),
        by_level(panel.top_turnover),
        by_level(panel.coverage),
        corrs,
    )


def summarize(
    panel: FactorPanel,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    horizons: Sequence[int] = HORIZONS,
) -> pd.DataFrame:
    ic, quint, turnover, coverage, _ = _slice(panel, start, end)
    if ic.empty:
        return pd.DataFrame()
    rows = []
    for name, group in ic.groupby(level="factor", sort=False):
        s1 = group[f"ic_{horizons[0]}m"].dropna()
        if len(s1) < 6:
            continue
        row: dict[str, object] = {"factor": name, "months": len(s1)}
        for h in horizons:
            s = group[f"ic_{h}m"].dropna()
            row[f"mean_ic_{h}m"] = s.mean() if len(s) else np.nan
            # Monthly observations of an h-month return overlap h-1 months.
            row[f"nw_t_{h}m"] = newey_west_t(s, lags=max(h, 1)) if len(s) > 2 else np.nan
        row["ic_ir"] = s1.mean() / s1.std(ddof=1) * math.sqrt(12) if s1.std(ddof=1) > 0 else np.nan
        row["hit_rate"] = float((s1 > 0).mean())
        half = len(s1) // 2
        row["ic_first_half"] = s1.iloc[:half].mean()
        row["ic_second_half"] = s1.iloc[half:].mean()

        if not quint.empty and name in quint.index.get_level_values("factor"):
            q = quint.xs(name, level="factor")
            spread = (q["Q5"] - q["Q1"]).dropna()
            row["q5_q1_annual"] = spread.mean() * 12
            row["q5_q1_nw_t"] = newey_west_t(spread, lags=1)
            row["top_excess_annual"] = q["Q5"].mean() * 12
            means = q[["Q1", "Q2", "Q3", "Q4", "Q5"]].mean()
            # Spearman of quintile number vs mean return: +1 = perfectly monotonic.
            row["monotonicity"] = float(
                pd.Series(np.arange(5.0)).corr(pd.Series(means.rank().to_numpy()))
            )
        if name in turnover.columns:
            row["top_quintile_turnover"] = turnover[name].mean()
        if name in coverage.columns:
            row["coverage"] = coverage[name].mean()
        rows.append(row)

    out = pd.DataFrame(rows).set_index("factor")
    t = out[f"nw_t_{horizons[0]}m"]
    out["p_value"] = t.map(two_sided_p)
    out["q_value_bh"] = benjamini_hochberg(out["p_value"])
    return out


def average_correlation(panel: FactorPanel, start=None, end=None) -> pd.DataFrame:
    corrs = _slice(panel, start, end)[4]
    if not corrs:
        return pd.DataFrame()
    stacked = pd.concat(corrs, keys=range(len(corrs)))
    return stacked.groupby(level=1).mean().reindex(index=corrs[-1].index, columns=corrs[-1].columns)


def ic_by_year(panel: FactorPanel, horizon: int = 1) -> pd.DataFrame:
    if panel.ic.empty:
        return pd.DataFrame()
    s = panel.ic[f"ic_{horizon}m"].reset_index()
    s["year"] = pd.DatetimeIndex(s["date"]).year
    return s.pivot_table(index="factor", columns="year", values=f"ic_{horizon}m", aggfunc="mean")


# --------------------------------------------------------------------------- driver
@dataclass
class ResearchReport:
    research: pd.DataFrame
    holdout: pd.DataFrame
    verdicts: pd.DataFrame
    correlations: pd.DataFrame
    ic_yearly: pd.DataFrame
    composite: dict[str, dict[str, float]]
    accepted: list[str]
    metadata: dict[str, object] = field(default_factory=dict)
    panel: FactorPanel | None = field(default=None, repr=False)


def _verdicts(
    research: pd.DataFrame,
    holdout: pd.DataFrame,
    candidates: Sequence[Candidate],
    fdr: float,
) -> pd.DataFrame:
    meta = {c.name: c for c in candidates}
    rows = []
    for name, r in research.iterrows():
        t = r.get("nw_t_1m", np.nan)
        q = r.get("q_value_bh", np.nan)
        positive = r.get("mean_ic_1m", 0) > 0
        significant = bool(np.isfinite(q) and q < fdr)
        h_ic = holdout["mean_ic_1m"].get(name, np.nan) if not holdout.empty else np.nan
        h_t = holdout["nw_t_1m"].get(name, np.nan) if not holdout.empty else np.nan
        holdout_agrees = bool(np.isfinite(h_t) and h_ic > 0 and h_t >= HOLDOUT_T)
        if significant and positive and holdout_agrees:
            verdict = "ACCEPT"
        elif significant and positive:
            verdict = "FAILS_HOLDOUT"
        elif significant:
            verdict = "INVERTED"  # predicts, but against the hypothesis: do not flip silently
        else:
            verdict = "NOT_SIGNIFICANT"
        rows.append(
            {
                "factor": name,
                "family": meta[name].family if name in meta else "",
                "higher_is_better": meta[name].higher_is_better if name in meta else None,
                "in_strategy": name in IN_STRATEGY,
                "mean_ic_1m": r.get("mean_ic_1m"),
                "nw_t_1m": t,
                "q_value_bh": q,
                "passes_t3": bool(np.isfinite(t) and t > T_HURDLE),
                "holdout_mean_ic_1m": h_ic,
                "holdout_nw_t_1m": h_t,
                "verdict": verdict,
                "hypothesis": meta[name].hypothesis if name in meta else "",
            }
        )
    order = {"ACCEPT": 0, "FAILS_HOLDOUT": 1, "INVERTED": 2, "NOT_SIGNIFICANT": 3}
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_o"] = df["verdict"].map(order)
    return (
        df.sort_values(["_o", "nw_t_1m"], ascending=[True, False])
        .drop(columns="_o")
        .set_index("factor")
    )


def _composite_ic(
    market: MarketData,
    config: StrategyConfig,
    provider: FeatureProvider,
    candidates: Sequence[Candidate],
    accepted: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp | str,
) -> FactorPanel:
    chosen = [c for c in candidates if c.name in accepted]

    def with_composite(date: pd.Timestamp) -> pd.DataFrame:
        snap = provider(date)
        eligible = apply_eligibility(snap, config)["eligible"].to_numpy()
        frame = snap[eligible].copy()
        parts = [
            winsorized_sector_percentile(frame, c.feature, higher_is_better=c.higher_is_better)
            for c in chosen
            if c.feature in frame.columns
        ]
        composite = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
        out = snap.copy()
        out["research_composite"] = np.nan
        out.loc[frame.index, "research_composite"] = composite
        return out

    candidate = Candidate("research_composite", True, "composite", "equal-weight accepted factors")
    return build_panel(market, config, start, end, [candidate], provider=with_composite)


def research_factors(
    market: MarketData,
    config: StrategyConfig,
    start: str,
    end: str,
    holdout_start: str,
    candidates: Sequence[Candidate] = CANDIDATES,
    provider: FeatureProvider | None = None,
    fdr: float = 0.10,
    progress: Callable[[str], None] | None = None,
) -> ResearchReport:
    cache: dict[pd.Timestamp, pd.DataFrame] = {}
    base = provider or (lambda d: build_feature_snapshot(market, d))

    def cached(date: pd.Timestamp) -> pd.DataFrame:
        key = pd.Timestamp(date)
        if key not in cache:
            cache[key] = base(key)
        return cache[key]

    split = pd.Timestamp(holdout_start)
    panel = build_panel(market, config, start, end, candidates, cached, progress=progress)
    research = summarize(panel, end=split)
    holdout = summarize(panel, start=split)
    if research.empty:
        raise ValueError("no research observations: check the date range and data coverage")
    verdicts = _verdicts(research, holdout, candidates, fdr)
    accepted = list(verdicts.index[verdicts["verdict"] == "ACCEPT"])
    # Composite members are chosen on the research sample only (ACCEPT or
    # FAILS_HOLDOUT), so the composite's holdout IC remains a clean test.
    research_pass = list(verdicts.index[verdicts["verdict"].isin(["ACCEPT", "FAILS_HOLDOUT"])])
    composite: dict[str, dict[str, float]] = {}
    if research_pass:
        comp = _composite_ic(
            market, config, cached, candidates, research_pass, pd.Timestamp(start), end
        )
        for label, s, e in (("research", None, split), ("holdout", split, None)):
            summary = summarize(comp, start=s, end=e)
            if not summary.empty:
                composite[label] = {
                    k: float(v)
                    for k, v in summary.iloc[0].items()
                    if isinstance(v, (int, float, np.floating))
                }
    return ResearchReport(
        research=research,
        holdout=holdout,
        verdicts=verdicts,
        correlations=average_correlation(panel, end=split),
        ic_yearly=ic_by_year(panel),
        composite=composite,
        accepted=accepted,
        metadata={
            "start": start,
            "end": end,
            "holdout_start": holdout_start,
            "fdr": fdr,
            "t_hurdle": T_HURDLE,
            "holdout_t": HOLDOUT_T,
            "n_candidates_tested": int(research["p_value"].notna().sum()),
            "composite_members": research_pass,
            "months": len(panel.dates),
            "data_version": market.data_version,
            "data_source": market.metadata.get("source", "synthetic")
            if market.metadata
            else "synthetic",
        },
        panel=panel,
    )


def pairs_above(corr: pd.DataFrame, threshold: float = 0.7) -> list[tuple[str, str, float]]:
    """Highly correlated candidate pairs (redundant information)."""
    out = []
    for a, b in itertools.combinations(corr.columns, 2):
        v = corr.loc[a, b]
        if np.isfinite(v) and abs(v) >= threshold:
            out.append((a, b, float(v)))
    return sorted(out, key=lambda x: -abs(x[2]))
