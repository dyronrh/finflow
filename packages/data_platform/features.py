"""Point-in-time feature snapshots (README §7.3 ``factor_features_daily``).

``build_feature_snapshot(market, as_of)`` only reads prices up to the close of
``as_of`` and filings / estimates / share counts with ``available_at <= as_of``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_platform.market import MarketData
from data_platform.point_in_time import as_of_view, latest_as_of

TRADING_DAYS_MONTH = 21
TRADING_DAYS_YEAR = 252
FLOW_COLUMNS = ["revenue", "gross_profit", "ebitda", "net_income", "free_cash_flow"]


def build_feature_snapshot(market: MarketData, as_of: pd.Timestamp) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of)
    close = market.close.loc[:as_of]
    volume = market.volume.loc[:as_of]
    # Levels (market cap, $ volume, price filter) use the traded price; returns
    # use the dividend-adjusted series. See data_platform.sources.yahoo.
    traded = market.price_close.loc[:as_of] if market.price_close is not None else close
    if close.empty:
        raise ValueError(f"no price history on or before {as_of.date()}")
    decision_date = close.index[-1]

    out = market.securities[["security_id", "ticker", "sector_id", "industry_id"]].copy()
    out = out.set_index("security_id")

    # A security must have traded on the decision date; stale prices are not used.
    price = traded.iloc[-1].reindex(out.index)
    out["price"] = price
    out["market_cap_usd"] = price * _shares_as_of(market, as_of, out.index)
    out["adv_usd"] = (traded.tail(20) * volume.tail(20)).mean()

    log_ret = np.log(close.tail(64)).diff()
    out["volatility_63d"] = log_ret.std() * np.sqrt(TRADING_DAYS_YEAR)
    out["mom_12_1"] = _lagged_return(close, TRADING_DAYS_MONTH, TRADING_DAYS_YEAR)
    out["mom_6m"] = _lagged_return(close, 0, 6 * TRADING_DAYS_MONTH)
    out["mom_3m"] = _lagged_return(close, 0, 3 * TRADING_DAYS_MONTH)

    # Research candidates (not used by v0.1/v0.2 scoring; see packages/research).
    year = close.tail(TRADING_DAYS_YEAR + 1)
    year_ret = year.pct_change(fill_method=None).iloc[1:]
    enough = year_ret.notna().mean() >= 0.8
    out["volatility_252d"] = (year_ret.std() * np.sqrt(TRADING_DAYS_YEAR)).where(enough)
    out["beta_252d"] = _beta(year_ret, _benchmark_returns(market, year_ret.index)).where(enough)
    out["ret_1m"] = _lagged_return(close, 0, TRADING_DAYS_MONTH)
    out["high_52w_ratio"] = (close.iloc[-1] / year.max()).where(enough)
    out["log_market_cap"] = np.log(out["market_cap_usd"].where(out["market_cap_usd"] > 0))
    out["net_issuance_1y"] = _net_issuance(market, as_of, out.index)

    members = market.members_at(as_of)
    if members is not None:
        out["in_universe"] = out.index.isin(list(members))

    out = out.join(_fundamental_features(market, as_of, out["market_cap_usd"]))
    out = out.join(_revision_features(market, as_of))

    out = out.reset_index()
    out.insert(0, "as_of_date", decision_date)
    out["data_version"] = market.data_version
    return out


def _benchmark_returns(market: MarketData, index: pd.Index) -> pd.Series:
    if market.benchmark_close is not None:
        return market.benchmark_close.reindex(index).pct_change(fill_method=None)
    window = market.close.reindex(index)
    return window.pct_change(fill_method=None).mean(axis=1)


def _beta(returns: pd.DataFrame, benchmark: pd.Series) -> pd.Series:
    b = benchmark.reindex(returns.index)
    ok = b.notna()
    r = returns[ok].fillna(0.0)
    b = b[ok]
    if len(b) < 20 or b.var() == 0:
        return pd.Series(np.nan, index=returns.columns)
    rc = r - r.mean()
    bc = b - b.mean()
    return (rc.mul(bc, axis=0).sum() / (bc**2).sum()).reindex(returns.columns)


def _net_issuance(market: MarketData, as_of: pd.Timestamp, index: pd.Index) -> pd.Series:
    """Change in share count over ~1 year (buybacks < 0 < issuance), PIT."""
    if market.shares is None or market.shares.empty:
        return pd.Series(np.nan, index=index)
    shares = market.shares.dropna(subset=["shares_outstanding"])
    now = latest_as_of(shares, as_of).set_index("security_id")["shares_outstanding"]
    then = latest_as_of(shares, as_of - pd.Timedelta(days=365)).set_index("security_id")[
        "shares_outstanding"
    ]
    change = now / then.reindex(now.index).where(lambda v: v > 0) - 1.0
    return change.reindex(index)


def _shares_as_of(market: MarketData, as_of: pd.Timestamp, index: pd.Index) -> pd.Series:
    if market.shares is not None and not market.shares.empty:
        latest = latest_as_of(market.shares.dropna(subset=["shares_outstanding"]), as_of)
        return latest.set_index("security_id")["shares_outstanding"].reindex(index)
    if "shares_outstanding" in market.securities.columns:
        return market.securities.set_index("security_id")["shares_outstanding"].reindex(index)
    return pd.Series(np.nan, index=index)


def _lagged_return(close: pd.DataFrame, skip: int, lookback: int) -> pd.Series:
    if len(close) <= lookback:
        return pd.Series(np.nan, index=close.columns)
    end = close.iloc[-1 - skip]
    begin = close.iloc[-1 - lookback]
    return end / begin - 1.0


def _fundamental_features(
    market: MarketData, as_of: pd.Timestamp, market_cap: pd.Series
) -> pd.DataFrame:
    visible = as_of_view(market.fundamentals, as_of)
    if visible.empty:
        return pd.DataFrame(index=market_cap.index)
    visible = visible.sort_values(["security_id", "event_time"])
    latest_end = visible.groupby("security_id")["event_time"].transform("max")
    age = (latest_end - visible["event_time"]).dt.days

    # Trailing twelve months = the four quarters ending in the last ~year.
    # Quarters ending > 300 days before the latest one are outside the window,
    # so a missing quarter leaves the TTM undefined rather than silently short.
    window = visible[age <= 300]
    ttm = window.groupby("security_id")[FLOW_COLUMNS].sum(min_count=4)
    n_quarters = window.groupby("security_id").size()
    ttm = ttm[n_quarters.reindex(ttm.index) == 4]

    latest = visible[age == 0].groupby("security_id").tail(1).set_index("security_id")
    year_ago = (
        visible[(age >= 330) & (age <= 400)].groupby("security_id").tail(1).set_index("security_id")
    )

    mcap = market_cap.reindex(ttm.index)
    net_debt = latest["net_debt"].reindex(ttm.index)
    invested = latest["invested_capital"].reindex(ttm.index)
    ebitda = ttm["ebitda"].where(ttm["ebitda"] > 0)
    revenue = ttm["revenue"].where(ttm["revenue"] > 0)

    feats = pd.DataFrame(index=ttm.index)
    feats["earnings_yield"] = ttm["net_income"] / mcap
    feats["fcf_yield"] = ttm["free_cash_flow"] / mcap
    feats["ev_ebitda"] = (mcap + net_debt) / ebitda
    feats["roic"] = ttm["net_income"] / invested.where(invested > 0)
    feats["gross_margin"] = ttm["gross_profit"] / revenue
    feats["fcf_margin"] = ttm["free_cash_flow"] / revenue
    feats["net_debt_ebitda"] = net_debt / ebitda

    prev = year_ago.reindex(latest.index)
    prev_rev = prev["revenue"].where(prev["revenue"] > 0)
    feats["revenue_growth"] = (latest["revenue"] / prev_rev - 1.0).reindex(feats.index)
    prev_ni = prev["net_income"].where(prev["net_income"] > 0)
    feats["eps_growth"] = (latest["net_income"] / prev_ni - 1.0).reindex(feats.index)
    # Research candidates.
    positive_ic = invested.where(invested > 0)
    feats["sales_yield"] = ttm["revenue"] / mcap
    feats["gross_profitability"] = ttm["gross_profit"] / positive_ic
    feats["accruals"] = (ttm["net_income"] - ttm["free_cash_flow"]) / positive_ic
    prev_ic = prev["invested_capital"].where(prev["invested_capital"] > 0)
    feats["asset_growth"] = (latest["invested_capital"] / prev_ic - 1.0).reindex(feats.index)
    two_years = visible[age <= 650].copy()
    rev = two_years["revenue"].where(two_years["revenue"] > 0)
    two_years["gm_q"] = two_years["gross_profit"] / rev
    two_years["nm_q"] = two_years["net_income"] / rev
    grouped = two_years.groupby("security_id")
    n_q = grouped.size()
    feats["gross_margin_volatility"] = grouped["gm_q"].std().where(n_q >= 6).reindex(feats.index)
    feats["earnings_volatility"] = grouped["nm_q"].std().where(n_q >= 6).reindex(feats.index)

    feats["fundamentals_available_at"] = latest["available_at"].reindex(feats.index)
    return feats.replace([np.inf, -np.inf], np.nan)


def _revision_features(market: MarketData, as_of: pd.Timestamp) -> pd.DataFrame:
    estimates = market.estimates
    if estimates is None or estimates.empty:
        return pd.DataFrame()
    current = latest_as_of(estimates, as_of).set_index("security_id")
    if current.empty:
        return pd.DataFrame()
    feats = pd.DataFrame(index=current.index)
    for days, name in ((30, "eps_rev_30d"), (90, "eps_rev_90d")):
        past = latest_as_of(estimates, as_of - pd.Timedelta(days=days)).set_index("security_id")
        feats[name] = (
            current["fy1_eps_estimate"] / past["fy1_eps_estimate"].reindex(current.index) - 1.0
        )
    window = as_of_view(estimates, as_of)
    window = window[window["available_at"] > as_of - pd.Timedelta(days=30)]
    counts = window.groupby("security_id")[["n_revisions_up", "n_revisions_down"]].sum()
    total = counts.sum(axis=1).replace(0, np.nan)
    feats["revisions_up_ratio"] = (counts["n_revisions_up"] / total).reindex(feats.index)
    feats["estimates_available_at"] = current["available_at"]
    return feats
