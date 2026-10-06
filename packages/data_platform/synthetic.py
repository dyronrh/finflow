"""Deterministic synthetic market for local development and tests.

It mimics the shape of real vendor data (daily OHLCV, quarterly filings with
a publication lag, weekly analyst-estimate snapshots) so the whole pipeline
can run end to end without a data licence. It is **not** market data and any
backtest on it says nothing about real-world performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_platform.market import MarketData

SECTORS = (
    "information_technology",
    "health_care",
    "financials",
    "consumer_discretionary",
    "industrials",
    "communication_services",
    "consumer_staples",
    "energy",
)


# Backwards-compatible name: the synthetic generator returns a plain MarketData.
SyntheticMarket = MarketData


def generate_synthetic_market(
    n_securities: int = 150,
    start: str = "2019-01-01",
    end: str = "2026-10-05",
    seed: int = 42,
) -> SyntheticMarket:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n_days = len(dates)
    ids = [f"SEC{i:04d}" for i in range(n_securities)]

    sector_idx = rng.integers(0, len(SECTORS), n_securities)
    sectors = np.array(SECTORS)[sector_idx]
    industries = np.array(
        [f"{s}_{k}" for s, k in zip(sectors, rng.integers(0, 2, n_securities), strict=True)]
    )

    # Latent characteristics drive both fundamentals and (weakly) returns.
    quality = rng.normal(0, 1, n_securities)
    growth = rng.normal(0, 1, n_securities)
    revision = rng.normal(0, 1, n_securities)
    idio_vol = rng.uniform(0.15, 0.55, n_securities)

    alpha = (0.02 * quality + 0.015 * growth + 0.02 * revision) / 252
    market = rng.normal(0.0003, 0.010, n_days)
    sector_shocks = rng.normal(0, 0.006, (n_days, len(SECTORS)))
    idio = rng.normal(0, 1, (n_days, n_securities)) * (idio_vol / np.sqrt(252))
    log_ret = market[:, None] + sector_shocks[:, sector_idx] + idio + alpha[None, :]
    log_ret[0] = 0.0

    start_price = np.exp(rng.normal(np.log(60), 0.8, n_securities))
    close = start_price * np.exp(np.cumsum(log_ret, axis=0))
    gap = rng.normal(0, 0.004, (n_days, n_securities))
    open_ = np.vstack([close[:1], close[:-1]]) * np.exp(gap)

    target_mcap = np.exp(rng.normal(np.log(12e9), 1.1, n_securities))
    shares = target_mcap / start_price
    target_adv = np.exp(rng.normal(np.log(40e6), 1.0, n_securities))
    volume = (target_adv / close) * np.exp(rng.normal(0, 0.3, (n_days, n_securities)))

    securities = pd.DataFrame(
        {
            "security_id": ids,
            "ticker": [f"T{i:03d}" for i in range(n_securities)],
            "sector_id": sectors,
            "industry_id": industries,
            "shares_outstanding": shares,
        }
    )

    fundamentals = _fundamentals(rng, ids, dates, quality, growth, target_mcap)
    estimates = _estimates(rng, ids, dates, quality, revision)

    return SyntheticMarket(
        securities=securities,
        open=pd.DataFrame(open_, index=dates, columns=ids),
        close=pd.DataFrame(close, index=dates, columns=ids),
        volume=pd.DataFrame(volume, index=dates, columns=ids),
        fundamentals=fundamentals,
        estimates=estimates,
        seed=seed,
    )


def _fundamentals(
    rng: np.random.Generator,
    ids: list[str],
    dates: pd.DatetimeIndex,
    quality: np.ndarray,
    growth: np.ndarray,
    target_mcap: np.ndarray,
) -> pd.DataFrame:
    period_ends = pd.date_range(dates[0] - pd.DateOffset(years=2), dates[-1], freq="QE")
    rows = []
    for i, security_id in enumerate(ids):
        revenue = target_mcap[i] / 4 / np.exp(rng.normal(np.log(3), 0.4))
        quarterly_growth = 0.015 + 0.012 * growth[i]
        gross_margin = np.clip(0.40 + 0.10 * quality[i], 0.05, 0.85)
        ebitda_margin = np.clip(0.18 + 0.07 * quality[i], -0.10, 0.60)
        net_margin = ebitda_margin - 0.07
        fcf_margin = net_margin + rng.normal(0.01, 0.02)
        invested_capital = revenue * 4 * np.exp(rng.normal(0, 0.3))
        leverage = np.clip(rng.normal(1.5 - 0.5 * quality[i], 0.8), -1.0, 6.0)
        for period_end in period_ends:
            revenue *= np.exp(quarterly_growth + rng.normal(0, 0.03))
            noise = rng.normal(0, 0.01, 4)
            ebitda = revenue * (ebitda_margin + noise[0])
            lag_days = int(rng.integers(25, 60))
            available_at = period_end + pd.Timedelta(days=lag_days)
            rows.append(
                {
                    "security_id": security_id,
                    "event_time": period_end,
                    "available_at": available_at,
                    "ingested_at": available_at + pd.Timedelta(days=1),
                    "revenue": revenue,
                    "gross_profit": revenue * (gross_margin + noise[1]),
                    "ebitda": ebitda,
                    "net_income": revenue * (net_margin + noise[2]),
                    "free_cash_flow": revenue * (fcf_margin + noise[3]),
                    "invested_capital": invested_capital,
                    "net_debt": leverage * ebitda * 4,
                }
            )
    return pd.DataFrame(rows)


def _estimates(
    rng: np.random.Generator,
    ids: list[str],
    dates: pd.DatetimeIndex,
    quality: np.ndarray,
    revision: np.ndarray,
) -> pd.DataFrame:
    snapshot_dates = pd.date_range(dates[0] - pd.DateOffset(months=6), dates[-1], freq="W-FRI")
    n = len(snapshot_dates)
    frames = []
    for i, security_id in enumerate(ids):
        drift = 0.002 * revision[i]
        log_eps = np.log(np.exp(rng.normal(1.0 + 0.3 * quality[i], 0.4))) + np.cumsum(
            rng.normal(drift, 0.01, n)
        )
        up_rate = np.clip(2.0 + 1.5 * revision[i], 0.2, None)
        down_rate = np.clip(2.0 - 1.5 * revision[i], 0.2, None)
        frames.append(
            pd.DataFrame(
                {
                    "security_id": security_id,
                    "event_time": snapshot_dates,
                    "available_at": snapshot_dates,
                    "fy1_eps_estimate": np.exp(log_eps),
                    "n_revisions_up": rng.poisson(up_rate, n),
                    "n_revisions_down": rng.poisson(down_rate, n),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
