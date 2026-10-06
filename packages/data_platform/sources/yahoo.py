"""Daily prices from Yahoo Finance via ``yfinance``.

Prices are split- and dividend-adjusted (``auto_adjust=True``), so close-to-close
returns approximate total returns. Volume from Yahoo is split-adjusted, so
``adjusted close × volume`` approximates traded dollar volume.

Yahoo is *not* used for fundamentals: it only exposes the last few quarters
and no publication dates, which would break point-in-time discipline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

FIELDS = ("Open", "Close", "Volume")


@dataclass
class YahooPrices:
    open: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    splits: pd.DataFrame
    """Long format: security_id, date, ratio (e.g. 4.0 for a 4-for-1 split)."""
    failed: list[str]


def to_yahoo_symbol(ticker: str) -> str:
    """``BRK.B`` → ``BRK-B`` (Yahoo uses dashes for share classes)."""
    return ticker.strip().upper().replace(".", "-")


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    chunk_size: int = 100,
    pause_seconds: float = 1.0,
) -> YahooPrices:
    import yfinance as yf  # optional dependency: uv sync --extra real-data

    symbols = sorted({to_yahoo_symbol(t) for t in tickers})
    frames: dict[str, list[pd.DataFrame]] = {f: [] for f in (*FIELDS, "Stock Splits")}
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i : i + chunk_size]
        raw = yf.download(
            chunk,
            start=start,
            end=end,
            auto_adjust=True,
            actions=True,
            group_by="column",
            threads=True,
            progress=False,
        )
        if raw is None or raw.empty:
            continue
        for field in frames:
            if field in raw.columns.get_level_values(0):
                frames[field].append(_as_wide(raw[field], chunk))
        time.sleep(pause_seconds)

    def combine(field: str) -> pd.DataFrame:
        parts = frames[field]
        if not parts:
            return pd.DataFrame()
        wide = pd.concat(parts, axis=1)
        wide.index = pd.DatetimeIndex(wide.index).tz_localize(None).normalize()
        return wide.sort_index()

    close = combine("Close").dropna(axis=1, how="all")
    failed = sorted(set(symbols) - set(close.columns))
    splits_wide = combine("Stock Splits")
    splits = (
        splits_wide.stack()
        .rename("ratio")
        .reset_index()
        .set_axis(["date", "security_id", "ratio"], axis=1)
        if not splits_wide.empty
        else pd.DataFrame(columns=["date", "security_id", "ratio"])
    )
    splits = splits[splits["ratio"] > 0][["security_id", "date", "ratio"]].reset_index(drop=True)
    return YahooPrices(
        open=combine("Open").reindex(columns=close.columns),
        close=close,
        volume=combine("Volume").reindex(columns=close.columns),
        splits=splits,
        failed=failed,
    )


def download_benchmark(symbol: str, start: str, end: str) -> pd.Series:
    prices = download_prices([symbol], start, end)
    if prices.close.empty:
        raise RuntimeError(f"could not download benchmark {symbol}")
    return prices.close.iloc[:, 0].rename(symbol)


def _as_wide(frame: pd.DataFrame | pd.Series, chunk: list[str]) -> pd.DataFrame:
    # yfinance returns a Series-like single column when one symbol is requested.
    if isinstance(frame, pd.Series):
        return frame.to_frame(chunk[0])
    return frame


def split_factor_after(splits: pd.DataFrame, security_id: str, after: pd.Timestamp) -> float:
    """Product of split ratios strictly after ``after``.

    Used only to express an old share count in today's (split-adjusted) units so
    it matches adjusted prices; it does not leak any economic information.
    """
    s = splits[(splits["security_id"] == security_id) & (splits["date"] > pd.Timestamp(after))]
    return float(s["ratio"].prod()) if len(s) else 1.0
