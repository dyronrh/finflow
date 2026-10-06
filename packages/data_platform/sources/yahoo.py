"""Daily prices from Yahoo Finance via ``yfinance``.

Two price series are kept:

* ``open`` / ``close``: split- **and dividend**-adjusted, so returns are total
  returns. Used for momentum, volatility, fills and valuation.
* ``price_close``: split-adjusted only (the price that actually traded, in
  today's share units). Used for market cap, the minimum-price filter and
  dollar volume. Dividend-adjusted levels must not be used there: Yahoo lowers
  historical prices by dividends paid *later*, which would make future
  dividend payers look cheaper (a look-ahead bias in the value factor).

Yahoo is *not* used for fundamentals: it only exposes the last few quarters
and no publication dates, which would break point-in-time discipline.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class YahooPrices:
    open: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    splits: pd.DataFrame
    """Long format: security_id, date, ratio (e.g. 4.0 for a 4-for-1 split)."""
    failed: list[str]
    price_close: pd.DataFrame | None = None
    """Split-adjusted (not dividend-adjusted) close."""


def to_yahoo_symbol(ticker: str) -> str:
    """``BRK.B`` → ``BRK-B`` (Yahoo uses dashes for share classes)."""
    return ticker.strip().upper().replace(".", "-")


class YahooDownloadError(RuntimeError):
    """Yahoo could not be reached reliably (network, DNS, rate limit)."""


Downloader = Callable[[list[str], str, str], dict[str, pd.DataFrame]]


def _yf_downloader(threads: int, cache_dir: Path | None) -> Downloader:
    import yfinance as yf  # optional dependency: uv sync --extra real-data

    if cache_dir is not None:
        # yfinance keeps a SQLite timezone cache; the default location can be
        # unwritable or locked under concurrency ("unable to open database file").
        tz_dir = Path(cache_dir) / "yfinance_tz"
        tz_dir.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(Exception):
            yf.set_tz_cache_location(str(tz_dir))

    def download(chunk: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        raw = yf.download(
            chunk,
            start=start,
            end=end,
            auto_adjust=False,
            actions=True,
            group_by="ticker",
            threads=threads,
            progress=False,
        )
        return split_by_symbol(raw, chunk)

    return download


def split_by_symbol(raw: pd.DataFrame | None, chunk: list[str]) -> dict[str, pd.DataFrame]:
    """One OHLCV frame per symbol that actually returned prices."""
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    for sym in chunk:
        frame = None
        if isinstance(raw.columns, pd.MultiIndex):
            for level in range(raw.columns.nlevels):
                if sym in raw.columns.get_level_values(level):
                    frame = raw.xs(sym, axis=1, level=level)
                    break
        elif len(chunk) == 1:
            frame = raw
        if frame is None or "Close" not in frame.columns or frame["Close"].dropna().empty:
            continue
        frame = frame.dropna(how="all").copy()
        frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
        out[sym] = frame
    return out


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: Path | None = None,
    chunk_size: int = 20,
    threads: int = 4,
    pause_seconds: float = 1.0,
    max_rounds: int = 4,
    downloader: Downloader | None = None,
    sleep: Callable[[float], None] | None = None,
) -> YahooPrices:
    """Download in small chunks with limited concurrency, retrying failures.

    Successful symbols are cached one file per symbol (``cache_dir``), so an
    interrupted run resumes where it stopped. A symbol is only reported as
    failed after several rounds; it is cached as missing only when other
    symbols in the same chunk succeeded (a real "no data", not an outage).
    """
    fetch = downloader or _yf_downloader(threads, cache_dir)
    sleep = sleep or time.sleep
    symbols = sorted({to_yahoo_symbol(t) for t in tickers})
    store = Path(cache_dir) / "prices" / f"{start}_{end}" if cache_dir else None
    if store:
        store.mkdir(parents=True, exist_ok=True)

    data: dict[str, pd.DataFrame] = {}
    known_missing: set[str] = set()
    for sym in symbols:
        if store and (store / f"{sym}.parquet").exists():
            data[sym] = pd.read_parquet(store / f"{sym}.parquet")
        elif store and (store / f"{sym}.missing").exists():
            known_missing.add(sym)

    pending = [s for s in symbols if s not in data and s not in known_missing]
    # Symbols that failed while others in the same request succeeded: evidence
    # the connection worked, so the miss is real (delisted, unknown ticker).
    failed_while_healthy: set[str] = set()
    for round_ in range(max_rounds):
        if not pending:
            break
        if round_:
            wait = pause_seconds * 15 * 2 ** (round_ - 1)
            log.warning(
                "Yahoo: retrying %d symbols in %.0fs (round %d/%d)",
                len(pending),
                wait,
                round_ + 1,
                max_rounds,
            )
            sleep(wait)
        still_missing: list[str] = []
        for i in range(0, len(pending), chunk_size):
            chunk = pending[i : i + chunk_size]
            try:
                got = fetch(chunk, start, end)
            except Exception as exc:  # network errors surface as exceptions too
                log.warning("Yahoo chunk failed (%s)", exc)
                got = {}
            for sym, frame in got.items():
                data[sym] = frame
                if store:
                    frame.to_parquet(store / f"{sym}.parquet")
            missing = [s for s in chunk if s not in got]
            if got:
                failed_while_healthy.update(missing)
            still_missing.extend(missing)
            sleep(pause_seconds)
        pending = still_missing
        log.info("Yahoo: %d/%d symbols with prices", len(data), len(symbols))

    if store:
        for sym in set(pending) & failed_while_healthy:
            (store / f"{sym}.missing").touch()

    return assemble_prices(data, symbols)


def assemble_prices(data: dict[str, pd.DataFrame], symbols: list[str]) -> YahooPrices:
    def wide(field: str) -> pd.DataFrame:
        cols = {s: f[field] for s, f in data.items() if field in f.columns}
        if not cols:
            return pd.DataFrame()
        return pd.DataFrame(cols).sort_index()

    price_close = wide("Close").dropna(axis=1, how="all")
    adj_close = wide("Adj Close").reindex(columns=price_close.columns)
    if adj_close.empty:
        adj_close = price_close
    adj_close = adj_close.fillna(price_close)
    # Total-return adjustment factor, applied to the open as well.
    factor = (adj_close / price_close).where(price_close > 0)
    close = adj_close
    splits_wide = wide("Stock Splits").reindex(columns=close.columns)
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
        open=wide("Open").reindex(index=close.index, columns=close.columns) * factor,
        close=close,
        price_close=price_close,
        volume=wide("Volume").reindex(index=close.index, columns=close.columns),
        splits=splits,
        failed=sorted(set(symbols) - set(close.columns)),
    )


def download_benchmark(
    symbol: str, start: str, end: str, cache_dir: Path | None = None
) -> pd.Series:
    prices = download_prices([symbol], start, end, cache_dir=cache_dir)
    if prices.close.empty:
        raise YahooDownloadError(
            f"could not download {symbol} from Yahoo Finance. This is a network/rate-limit "
            "problem, not missing data: check your connection and re-run `make fetch-data` "
            "(prices already downloaded are cached)."
        )
    return prices.close.iloc[:, 0].rename(symbol)


def split_factor_after(splits: pd.DataFrame, security_id: str, after: pd.Timestamp) -> float:
    """Product of split ratios strictly after ``after``.

    Used only to express an old share count in today's (split-adjusted) units so
    it matches adjusted prices; it does not leak any economic information.
    """
    s = splits[(splits["security_id"] == security_id) & (splits["date"] > pd.Timestamp(after))]
    return float(s["ratio"].prod()) if len(s) else 1.0
