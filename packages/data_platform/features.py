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
    if close.empty:
        raise ValueError(f"no price history on or before {as_of.date()}")
    decision_date = close.index[-1]

    out = market.securities[["security_id", "ticker", "sector_id", "industry_id"]].copy()
    out = out.set_index("security_id")

    # A security must have traded on the decision date; stale prices are not used.
    price = close.iloc[-1]
    out["price"] = price
    out["market_cap_usd"] = price * _shares_as_of(market, as_of, out.index)
    out["adv_usd"] = (close.tail(20) * volume.tail(20)).mean()

    log_ret = np.log(close.tail(64)).diff()
    out["volatility_63d"] = log_ret.std() * np.sqrt(TRADING_DAYS_YEAR)
    out["mom_12_1"] = _lagged_return(close, TRADING_DAYS_MONTH, TRADING_DAYS_YEAR)
    out["mom_6m"] = _lagged_return(close, 0, 6 * TRADING_DAYS_MONTH)
    out["mom_3m"] = _lagged_return(close, 0, 3 * TRADING_DAYS_MONTH)

    members = market.members_at(as_of)
    if members is not None:
        out["in_universe"] = out.index.isin(list(members))

    out = out.join(_fundamental_features(market, as_of, out["market_cap_usd"]))
    out = out.join(_revision_features(market, as_of))

    out = out.reset_index()
    out.insert(0, "as_of_date", decision_date)
    out["data_version"] = market.data_version
    return out


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
