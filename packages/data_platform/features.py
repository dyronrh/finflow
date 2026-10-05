"""Point-in-time feature snapshots (README §7.3 ``factor_features_daily``).

``build_feature_snapshot(market, as_of)`` only reads prices up to the close of
``as_of`` and filings / estimates with ``available_at <= as_of``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_platform.point_in_time import as_of_view, latest_as_of
from data_platform.synthetic import SyntheticMarket

TRADING_DAYS_MONTH = 21
TRADING_DAYS_YEAR = 252


def build_feature_snapshot(market: SyntheticMarket, as_of: pd.Timestamp) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of)
    close = market.close.loc[:as_of]
    volume = market.volume.loc[:as_of]
    if close.empty:
        raise ValueError(f"no price history on or before {as_of.date()}")
    decision_date = close.index[-1]

    out = market.securities[["security_id", "ticker", "sector_id", "industry_id"]].copy()
    out = out.set_index("security_id")

    price = close.iloc[-1]
    out["price"] = price
    out["market_cap_usd"] = price * market.securities.set_index("security_id")["shares_outstanding"]
    out["adv_usd"] = (close.tail(20) * volume.tail(20)).mean()

    log_ret = np.log(close).diff().tail(63)
    out["volatility_63d"] = log_ret.std() * np.sqrt(TRADING_DAYS_YEAR)
    out["mom_12_1"] = _lagged_return(close, TRADING_DAYS_MONTH, TRADING_DAYS_YEAR)
    out["mom_6m"] = _lagged_return(close, 0, 6 * TRADING_DAYS_MONTH)
    out["mom_3m"] = _lagged_return(close, 0, 3 * TRADING_DAYS_MONTH)

    out = out.join(_fundamental_features(market, as_of, out["market_cap_usd"]))
    out = out.join(_revision_features(market, as_of))

    out = out.reset_index()
    out.insert(0, "as_of_date", decision_date)
    out["data_version"] = market.data_version
    return out


def _lagged_return(close: pd.DataFrame, skip: int, lookback: int) -> pd.Series:
    if len(close) <= lookback:
        return pd.Series(np.nan, index=close.columns)
    end = close.iloc[-1 - skip]
    begin = close.iloc[-1 - lookback]
    return end / begin - 1.0


def _fundamental_features(
    market: SyntheticMarket, as_of: pd.Timestamp, market_cap: pd.Series
) -> pd.DataFrame:
    visible = as_of_view(market.fundamentals, as_of)
    if visible.empty:
        return pd.DataFrame(index=market_cap.index)
    ranked = visible.sort_values(["security_id", "event_time"], ascending=[True, False])
    ranked["recency"] = ranked.groupby("security_id").cumcount()
    latest = ranked[ranked["recency"] == 0].set_index("security_id")
    year_ago = ranked[ranked["recency"] == 4].set_index("security_id")
    ttm = (
        ranked[ranked["recency"] < 4]
        .groupby("security_id")[
            ["revenue", "gross_profit", "ebitda", "net_income", "free_cash_flow"]
        ]
        .sum()
    )
    full_year = ranked[ranked["recency"] < 4].groupby("security_id").size() == 4
    ttm = ttm[full_year.reindex(ttm.index, fill_value=False)]

    mcap = market_cap.reindex(ttm.index)
    net_debt = latest["net_debt"].reindex(ttm.index)
    ebitda = ttm["ebitda"].where(ttm["ebitda"] > 0)
    feats = pd.DataFrame(index=ttm.index)
    feats["earnings_yield"] = ttm["net_income"] / mcap
    feats["fcf_yield"] = ttm["free_cash_flow"] / mcap
    feats["ev_ebitda"] = (mcap + net_debt) / ebitda
    feats["roic"] = ttm["net_income"] / latest["invested_capital"].reindex(ttm.index)
    feats["gross_margin"] = ttm["gross_profit"] / ttm["revenue"]
    feats["fcf_margin"] = ttm["free_cash_flow"] / ttm["revenue"]
    feats["net_debt_ebitda"] = net_debt / ebitda

    prev = year_ago.reindex(latest.index)
    feats["revenue_growth"] = (latest["revenue"] / prev["revenue"] - 1.0).reindex(feats.index)
    prev_ni = prev["net_income"].where(prev["net_income"] > 0)
    feats["eps_growth"] = (latest["net_income"] / prev_ni - 1.0).reindex(feats.index)
    feats["fundamentals_available_at"] = latest["available_at"].reindex(feats.index)
    return feats


def _revision_features(market: SyntheticMarket, as_of: pd.Timestamp) -> pd.DataFrame:
    estimates = market.estimates
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
