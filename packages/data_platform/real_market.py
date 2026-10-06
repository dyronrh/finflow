"""Build a real US-equity ``MarketData`` from free sources.

* Universe: S&P 500 with approximate point-in-time membership (Wikipedia).
* Prices: Yahoo Finance daily OHLCV, split/dividend adjusted.
* Fundamentals and share counts: SEC EDGAR XBRL with filing dates.
* Benchmark: SPY from Yahoo Finance.
* Analyst estimates: none (no free point-in-time source); the revisions factor
  is therefore absent and its weight is redistributed by the scorer.

Raw downloads are cached under ``<cache>/raw`` (bronze, never overwritten);
the assembled dataset under ``<cache>/processed`` so later runs work offline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from data_platform.market import MarketData
from data_platform.sources import sec_edgar, sp500, yahoo
from data_platform.sources.sic import sic_to_sector

log = logging.getLogger(__name__)

DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "local_dev_only" / "real"
_TABLES = (
    "open",
    "close",
    "price_close",
    "volume",
    "fundamentals",
    "shares",
    "securities",
    "membership",
)


def fetch_real_market(
    start: str,
    end: str,
    user_agent: str,
    cache_dir: Path = DEFAULT_CACHE,
    max_securities: int | None = None,
    benchmark: str = "SPY",
) -> MarketData:
    raw = Path(cache_dir) / "raw"
    sec = sec_edgar.SecClient(user_agent, raw / "sec")

    constituents_table, changes_table = sp500.fetch_tables(user_agent)
    current = sp500.parse_constituents(constituents_table)
    universe_note = "approximate point-in-time membership from the Wikipedia change log"
    try:
        if changes_table is None:
            raise ValueError("change-log table not found on the Wikipedia page")
        changes = sp500.parse_changes(changes_table)
        if changes.empty:
            raise ValueError("change-log table parsed to zero dated rows")
    except ValueError as exc:
        log.warning("S&P 500 change log unusable (%s); using current members only", exc)
        changes = pd.DataFrame(columns=["date", "added", "removed"])
        universe_note = f"current members only ({exc}): strong survivorship bias"
    membership = sp500.membership_intervals(current["security_id"].tolist(), changes)
    membership = membership[(membership["end"].isna()) | (membership["end"] > pd.Timestamp(start))]

    tickers = sorted(set(membership["security_id"]))
    if max_securities:
        tickers = sorted(set(current["security_id"]))[:max_securities]
        membership = membership[membership["security_id"].isin(tickers)]

    log.info("downloading prices for %d tickers from Yahoo Finance", len(tickers))
    prices = yahoo.download_prices(tickers, start, end)
    spy = yahoo.download_benchmark(benchmark, start, end)
    have_prices = list(prices.close.columns)
    log.info("prices: %d ok, %d missing (mostly delisted)", len(have_prices), len(prices.failed))

    cik_map = sec.ticker_to_cik()
    cik_from_wiki = dict(zip(current["security_id"], current.get("cik", pd.Series()), strict=False))
    sectors = current.set_index("security_id")[["ticker", "sector_id", "industry_id"]]

    fundamentals, shares, securities, no_facts = [], [], [], []
    for i, sid in enumerate(have_prices):
        cik = cik_from_wiki.get(sid)
        cik = int(cik) if pd.notna(cik) else cik_map.get(sid)
        if cik is None:
            no_facts.append(sid)
            continue
        facts = sec.company_facts(cik)
        if not facts:
            no_facts.append(sid)
            continue
        f = sec_edgar.parse_company_facts(facts, sid)
        s = sec_edgar.parse_share_counts(facts, sid)
        if not s.empty:
            # Express historical share counts in today's split-adjusted units.
            s["shares_outstanding"] = [
                q * yahoo.split_factor_after(prices.splits, sid, filed)
                for q, filed in zip(s["shares_outstanding"], s["available_at"], strict=True)
            ]
        fundamentals.append(f)
        shares.append(s)

        if sid in sectors.index:
            row = sectors.loc[sid]
            sector, industry, ticker = row["sector_id"], row["industry_id"], row["ticker"]
        else:
            sic = sec.submissions(cik).get("sic")
            sector, industry, ticker = sic_to_sector(sic), f"sic_{sic}", sid
        securities.append(
            {"security_id": sid, "ticker": ticker, "sector_id": sector, "industry_id": industry}
        )
        if (i + 1) % 50 == 0:
            log.info("SEC facts: %d/%d", i + 1, len(have_prices))

    ids = [row["security_id"] for row in securities]
    market = MarketData(
        securities=pd.DataFrame(securities),
        open=prices.open[ids],
        close=prices.close[ids],
        price_close=prices.price_close[ids] if prices.price_close is not None else None,
        volume=prices.volume[ids],
        fundamentals=pd.concat(fundamentals, ignore_index=True),
        estimates=pd.DataFrame(),
        shares=pd.concat(shares, ignore_index=True),
        membership=membership[membership["security_id"].isin(ids)].reset_index(drop=True),
        benchmark_close=spy.reindex(prices.close.index).ffill(),
        metadata={
            "source": "yahoo_finance+sec_edgar+wikipedia_sp500",
            "survivorship_bias": (
                "partial: removed S&P 500 members without Yahoo history are missing"
            ),
            "universe": universe_note,
            "change_log_rows": len(changes),
            "tickers_requested": len(tickers),
            "tickers_without_prices": prices.failed,
            "tickers_without_sec_facts": no_facts,
            "analyst_estimates": "not available (revisions factor disabled)",
        },
    )
    save_market(market, Path(cache_dir) / "processed")
    return market


def save_market(market: MarketData, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in _TABLES:
        frame = getattr(market, name)
        if frame is not None:
            frame.to_parquet(directory / f"{name}.parquet")
    if market.benchmark_close is not None:
        market.benchmark_close.to_frame("benchmark").to_parquet(directory / "benchmark.parquet")
    meta = {**market.metadata, "data_version": market.data_version}
    (directory / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))


def load_market(directory: Path = DEFAULT_CACHE / "processed") -> MarketData:
    directory = Path(directory)
    if not (directory / "close.parquet").exists():
        raise FileNotFoundError(f"no cached real data in {directory}; run `make fetch-data` first")
    tables = {
        name: pd.read_parquet(directory / f"{name}.parquet")
        for name in _TABLES
        if (directory / f"{name}.parquet").exists()
    }
    meta = json.loads((directory / "metadata.json").read_text())
    warnings = list(meta.get("warnings", []))
    if "price_close" not in tables:
        warnings.append(
            "cache predates the dividend-adjustment fix: market cap and value factors use "
            "dividend-adjusted prices (look-ahead bias). Re-run `make fetch-data`."
        )
    if str(meta.get("universe", "")).startswith("current members only") or "universe" not in meta:
        warnings.append(
            "universe = current S&P 500 members only: strong survivorship bias, absolute "
            "returns are inflated."
        )
    meta["warnings"] = warnings
    for message in warnings:
        log.warning(message)
    bench_path = directory / "benchmark.parquet"
    return MarketData(
        securities=tables["securities"],
        open=tables["open"],
        close=tables["close"],
        price_close=tables.get("price_close"),
        volume=tables["volume"],
        fundamentals=tables["fundamentals"],
        estimates=pd.DataFrame(),
        shares=tables.get("shares"),
        membership=tables.get("membership"),
        benchmark_close=pd.read_parquet(bench_path)["benchmark"] if bench_path.exists() else None,
        data_version=meta.pop("data_version", ""),
        metadata=meta,
    )
