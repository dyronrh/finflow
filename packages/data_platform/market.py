"""Container for one point-in-time market dataset (synthetic or real)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class MarketData:
    securities: pd.DataFrame
    """security_id, ticker, sector_id, industry_id[, shares_outstanding]."""
    open: pd.DataFrame
    """Wide frame: index=date, columns=security_id. NaN where not trading."""
    close: pd.DataFrame
    volume: pd.DataFrame
    fundamentals: pd.DataFrame
    """Quarterly rows: security_id, event_time (period end), available_at, ingested_at,
    revenue, gross_profit, ebitda, net_income, free_cash_flow, invested_capital, net_debt."""
    estimates: pd.DataFrame
    """FY1 EPS estimate snapshots (may be empty: no free point-in-time source)."""
    seed: int | None = None
    data_version: str = ""
    shares: pd.DataFrame | None = None
    """Optional time-varying share counts: security_id, available_at, shares_outstanding."""
    membership: pd.DataFrame | None = None
    """Optional index membership intervals: security_id, start, end (end exclusive, NaT=open)."""
    high: pd.DataFrame | None = None
    """Optional daily highs/lows on the same (total-return adjusted) basis as open/close."""
    low: pd.DataFrame | None = None
    price_close: pd.DataFrame | None = None
    """Optional split-adjusted, *not* dividend-adjusted close for market cap, price
    filters and dollar volume. ``close`` is used when absent (synthetic data)."""
    benchmark_close: pd.Series | None = None
    """Optional external benchmark (e.g. SPY total-return proxy)."""
    equal_weight_reference: pd.Series | None = None
    """Optional survivorship-free equal-weight index proxy (e.g. RSP)."""
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.data_version:
            last = self.close.index[-1].strftime("%Y_%m_%d")
            prefix = "synthetic_us_equities" if self.seed is not None else "us_equities"
            suffix = f"_seed{self.seed}" if self.seed is not None else ""
            self.data_version = f"{prefix}{suffix}_{last}"

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close.index)

    def trading_date_on_or_before(self, as_of: pd.Timestamp) -> pd.Timestamp | None:
        dates = self.dates
        pos = dates.searchsorted(pd.Timestamp(as_of), side="right") - 1
        return None if pos < 0 else dates[pos]

    def display_ohlcv(self, security_id: str) -> pd.DataFrame:
        """OHLCV in traded units (split-adjusted, not dividend-adjusted) for charts."""
        close = self.close[security_id]
        traded = self.price_close[security_id] if self.price_close is not None else close
        factor = (traded / close).where(close > 0)
        opens = self.open[security_id]
        high = (
            self.high[security_id]
            if self.high is not None
            else pd.concat([opens, close], axis=1).max(axis=1)
        )
        low = (
            self.low[security_id]
            if self.low is not None
            else pd.concat([opens, close], axis=1).min(axis=1)
        )
        frame = pd.DataFrame(
            {
                "open": opens * factor,
                "high": high * factor,
                "low": low * factor,
                "close": traded,
                "volume": self.volume[security_id],
            }
        ).dropna(subset=["open", "close"])
        # Guard against vendor glitches: the bar must contain open and close.
        frame["high"] = frame[["high", "open", "close"]].max(axis=1)
        frame["low"] = frame[["low", "open", "close"]].min(axis=1)
        return frame

    def members_at(self, as_of: pd.Timestamp) -> set[str] | None:
        """Universe members at ``as_of`` or ``None`` when membership is not tracked."""
        if self.membership is None:
            return None
        m = self.membership
        as_of = pd.Timestamp(as_of)
        live = (m["start"].isna() | (m["start"] <= as_of)) & (m["end"].isna() | (m["end"] > as_of))
        return set(m.loc[live, "security_id"])
