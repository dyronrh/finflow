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
from data_platform.sources import sec_edgar, sp500, sp500_history, yahoo
from data_platform.sources.sic import sic_to_sector

log = logging.getLogger(__name__)

DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "local_dev_only" / "real"
_TABLES = (
    "open",
    "close",
    "price_close",
    "high",
    "low",
    "volume",
    "fundamentals",
    "shares",
    "securities",
    "membership",
)


def _membership(
    current: pd.DataFrame, changes_table: pd.DataFrame | None, user_agent: str
) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    """Point-in-time membership, best source first. Returns (intervals, note, changes)."""
    changes = pd.DataFrame(columns=["date", "added", "removed", "removed_name"])
    try:
        if changes_table is not None:
            changes = sp500.parse_changes(changes_table)
    except ValueError as exc:
        log.warning("Wikipedia change log unusable: %s", exc)

    current_ids = current["security_id"].tolist()
    try:
        long = sp500_history.parse_components(sp500_history.fetch_components_csv(user_agent))
        if long.empty:
            raise ValueError("historical components CSV parsed to zero rows")
        intervals = sp500_history.membership_from_snapshots(long)
        last = long["date"].max()
        intervals = sp500_history.reconcile_with_current(intervals, current_ids, last)
        note = (
            "point-in-time membership from fja05680/sp500 historical components "
            f"(through {last.date()}, reconciled with current constituents)"
        )
        return intervals, note, changes
    except Exception as exc:  # network, format change, rate limit...
        log.warning("historical components dataset unavailable (%s); trying Wikipedia", exc)

    if not changes.empty:
        intervals = sp500.membership_intervals(current_ids, changes)
        return (
            intervals,
            "approximate point-in-time membership from the Wikipedia change log",
            changes,
        )
    log.warning("no membership history available; using current members only")
    intervals = sp500.membership_intervals(current_ids, changes)
    return intervals, "current members only: strong survivorship bias", changes


def _resolve_cik(
    sid: str,
    is_current: bool,
    removed_name: str | None,
    wiki_cik: dict[str, object],
    ticker_cik: dict[str, int],
    ticker_title: dict[str, str],
    name_ciks: dict[str, set[int]] | None,
) -> tuple[int | None, str]:
    """CIK for a security, guarding against tickers recycled by other companies."""
    cik = wiki_cik.get(sid)
    if is_current and cik is not None and pd.notna(cik):
        return int(cik), "wikipedia"
    if is_current and sid in ticker_cik:
        return ticker_cik[sid], "sec_ticker"
    # Removed member: today's owner of the ticker may be a different company.
    if (
        removed_name
        and sid in ticker_cik
        and sec_edgar.names_match(ticker_title[sid], removed_name)
    ):
        return ticker_cik[sid], "sec_ticker_name_checked"
    if removed_name and name_ciks:
        matches = name_ciks.get(sec_edgar.normalize_company_name(removed_name), set())
        if len(matches) == 1:
            return next(iter(matches)), "sec_name_lookup"
    if sid in ticker_cik:
        # No name to check against; accepted only if filings overlap the
        # membership period (verified by the caller).
        return ticker_cik[sid], "sec_ticker_unverified"
    return None, "unresolved"


def _overlaps(fundamentals: pd.DataFrame, intervals: pd.DataFrame) -> bool:
    if fundamentals.empty:
        return False
    ends = fundamentals["event_time"]
    for row in intervals.itertuples():
        lo = row.start if pd.notna(row.start) else pd.Timestamp.min
        hi = row.end if pd.notna(row.end) else pd.Timestamp.max
        if ((ends >= lo) & (ends < hi)).any():
            return True
    return False


def coverage_by_year(market: MarketData) -> pd.DataFrame:
    """Share of index members (per year-end) with prices / with fundamentals."""
    if market.membership is None:
        return pd.DataFrame()
    with_prices = set(market.close.columns)
    with_facts = set(market.fundamentals["security_id"]) if len(market.fundamentals) else set()
    rows = []
    for year in sorted({d.year for d in market.dates}):
        at = pd.Timestamp(f"{year}-12-31")
        members = market.members_at(at) or set()
        if not members:
            continue
        rows.append(
            {
                "year": year,
                "members": len(members),
                "with_prices": len(members & with_prices) / len(members),
                "with_prices_and_fundamentals": len(members & with_prices & with_facts)
                / len(members),
            }
        )
    return pd.DataFrame(rows)


def fetch_real_market(
    start: str,
    end: str,
    user_agent: str,
    cache_dir: Path = DEFAULT_CACHE,
    max_securities: int | None = None,
    benchmark: str = "SPY",
    equal_weight_reference: str = "RSP",
) -> MarketData:
    raw = Path(cache_dir) / "raw"
    sec = sec_edgar.SecClient(user_agent, raw / "sec")

    constituents_table, changes_table = sp500.fetch_tables(user_agent)
    current = sp500.parse_constituents(constituents_table)
    membership, universe_note, changes = _membership(current, changes_table, user_agent)
    membership = membership[(membership["end"].isna()) | (membership["end"] > pd.Timestamp(start))]
    tagged = membership["security_id"].map(sp500_history.is_reused_ticker_tag)
    membership = membership[~tagged].reset_index(drop=True)
    log.info("universe: %s", universe_note)

    current_ids = set(current["security_id"])
    tickers = sorted(set(membership["security_id"]))
    if max_securities:
        tickers = sorted(current_ids)[:max_securities]
        membership = membership[membership["security_id"].isin(tickers)]
    log.info(
        "requesting %d tickers (%d current members, %d former members)",
        len(tickers),
        len(set(tickers) & current_ids),
        len(set(tickers) - current_ids),
    )

    yahoo_cache = raw / "yahoo"
    spy = yahoo.download_benchmark(benchmark, start, end, cache_dir=yahoo_cache)
    prices = yahoo.download_prices(tickers, start, end, cache_dir=yahoo_cache)
    missing_current = (current_ids & set(tickers)) - set(prices.close.columns)
    if len(missing_current) > 0.05 * max(len(current_ids & set(tickers)), 1):
        raise yahoo.YahooDownloadError(
            f"{len(missing_current)} current S&P 500 members got no prices from Yahoo "
            f"(e.g. {sorted(missing_current)[:5]}). Current members always trade, so this is "
            "a network or rate-limit problem. Re-run `make fetch-data` later: prices already "
            "downloaded are cached and will not be fetched again."
        )
    try:
        rsp = yahoo.download_benchmark(equal_weight_reference, start, end, cache_dir=yahoo_cache)
    except yahoo.YahooDownloadError as exc:
        log.warning("equal-weight reference unavailable: %s", exc)
        rsp = None

    # A former member whose Yahoo history starts after it left the index is a
    # different company that reused the ticker: drop it.
    first_price = prices.close.apply(pd.Series.first_valid_index)
    last_end = membership.groupby("security_id")["end"].max()
    has_open = membership.groupby("security_id")["end"].apply(lambda e: e.isna().any())
    reused = [
        sid
        for sid in prices.close.columns
        if not has_open.get(sid, True)
        and pd.notna(first_price.get(sid))
        and first_price[sid] >= last_end[sid]
    ]
    have_prices = [sid for sid in prices.close.columns if sid not in set(reused)]
    log.info(
        "prices: %d ok, %d missing (mostly delisted), %d dropped as recycled tickers",
        len(have_prices),
        len(prices.failed),
        len(reused),
    )

    ticker_cik = sec.ticker_to_cik()
    ticker_title = sec.ticker_titles()
    wiki_cik = dict(zip(current["security_id"], current.get("cik", pd.Series()), strict=False))
    removed_names = {
        r: n
        for r, n in zip(changes.get("removed", []), changes.get("removed_name", []), strict=False)
        if r and isinstance(n, str) and n and n.lower() != "nan"
    }
    name_ciks = None
    if any(sid not in current_ids for sid in have_prices):
        try:
            name_ciks = sec.name_to_ciks()
        except Exception as exc:
            log.warning("SEC name lookup unavailable (%s)", exc)
    sectors = current.set_index("security_id")[["ticker", "sector_id", "industry_id"]]

    fundamentals, shares, securities, no_facts, cik_sources = [], [], [], [], {}
    for i, sid in enumerate(have_prices):
        is_current = sid in current_ids
        cik, how = _resolve_cik(
            sid, is_current, removed_names.get(sid), wiki_cik, ticker_cik, ticker_title, name_ciks
        )
        facts = sec.company_facts(cik) if cik is not None else {}
        f = sec_edgar.parse_company_facts(facts, sid) if facts else pd.DataFrame()
        if f.empty or (
            not is_current and not _overlaps(f, membership[membership["security_id"] == sid])
        ):
            no_facts.append(sid)
            continue
        cik_sources[how] = cik_sources.get(how, 0) + 1
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
    former = [sid for sid in ids if sid not in current_ids]
    market = MarketData(
        securities=pd.DataFrame(securities),
        open=prices.open[ids],
        close=prices.close[ids],
        price_close=prices.price_close[ids] if prices.price_close is not None else None,
        high=prices.high[ids] if prices.high is not None else None,
        low=prices.low[ids] if prices.low is not None else None,
        volume=prices.volume[ids],
        fundamentals=pd.concat(fundamentals, ignore_index=True),
        estimates=pd.DataFrame(),
        shares=pd.concat(shares, ignore_index=True),
        # Membership keeps *all* members, including those without data, so that
        # coverage can be measured; the scorer only sees names with data anyway.
        membership=membership.reset_index(drop=True),
        benchmark_close=spy.reindex(prices.close.index).ffill(),
        equal_weight_reference=rsp.reindex(prices.close.index).ffill() if rsp is not None else None,
        metadata={
            "source": "yahoo_finance+sec_edgar+sp500_history",
            "universe": universe_note,
            "change_log_rows": len(changes),
            "members_in_window": int(membership["security_id"].nunique()),
            "tickers_requested": len(tickers),
            "former_members_with_data": len(former),
            "tickers_without_prices": prices.failed,
            "tickers_dropped_as_recycled": reused,
            "tickers_without_sec_facts": no_facts,
            "cik_resolution": cik_sources,
            "survivorship_bias": (
                "partial: former members without Yahoo prices or SEC facts are missing; "
                "compare the equal-weight universe with RSP to estimate the residual bias"
            ),
            "analyst_estimates": "not available (revisions factor disabled)",
        },
    )
    coverage = coverage_by_year(market)
    market.metadata["coverage_by_year"] = coverage.round(3).to_dict("records")
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
    if market.equal_weight_reference is not None:
        market.equal_weight_reference.to_frame("rsp").to_parquet(directory / "rsp.parquet")
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
        high=tables.get("high"),
        low=tables.get("low"),
        volume=tables["volume"],
        fundamentals=tables["fundamentals"],
        estimates=pd.DataFrame(),
        shares=tables.get("shares"),
        membership=tables.get("membership"),
        benchmark_close=pd.read_parquet(bench_path)["benchmark"] if bench_path.exists() else None,
        equal_weight_reference=(
            pd.read_parquet(directory / "rsp.parquet")["rsp"]
            if (directory / "rsp.parquet").exists()
            else None
        ),
        data_version=meta.pop("data_version", ""),
        metadata=meta,
    )
