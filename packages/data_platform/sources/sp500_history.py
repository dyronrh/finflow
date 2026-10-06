"""Historical S&P 500 composition from the public ``fja05680/sp500`` dataset.

The repository publishes a CSV named like
``S&P 500 Historical Components & Changes(MM-DD-YYYY).csv`` with one row per
composition change since 1996: ``date`` and ``tickers`` (comma-separated
members on that date). It is the primary membership source; the Wikipedia
change log is the fallback.
"""

from __future__ import annotations

import re
from io import StringIO

import pandas as pd

REPO_API = "https://api.github.com/repos/fja05680/sp500/contents"
FILE_PATTERN = re.compile(
    r"S&P 500 Historical Components & Changes\((\d{2})-(\d{2})-(\d{4})\)\.csv$"
)
# Some datasets tag a recycled ticker's earlier owner with a date suffix.
REUSED_SUFFIX = re.compile(r"-\d{6}$")


def normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper().replace(".", "-")


def fetch_components_csv(user_agent: str) -> pd.DataFrame:
    import requests

    headers = {"User-Agent": user_agent, "Accept": "application/vnd.github+json"}
    listing = requests.get(REPO_API, headers=headers, timeout=60)
    listing.raise_for_status()
    candidates = []
    for item in listing.json():
        match = FILE_PATTERN.search(item.get("name", ""))
        if match and item.get("download_url"):
            month, day, year = match.groups()
            candidates.append((f"{year}{month}{day}", item["download_url"]))
    if not candidates:
        raise ValueError("no 'Historical Components & Changes' CSV found in fja05680/sp500")
    _, url = max(candidates)
    response = requests.get(url, headers={"User-Agent": user_agent}, timeout=120)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def parse_components(table: pd.DataFrame) -> pd.DataFrame:
    """Long format: date, security_id (one row per member per change date)."""
    cols = {c.lower().strip(): c for c in table.columns}
    date_col = next((c for k, c in cols.items() if "date" in k), table.columns[0])
    tick_col = next((c for k, c in cols.items() if "ticker" in k), table.columns[1])
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(table[date_col], errors="coerce"),
            "tickers": table[tick_col].astype(str),
        }
    ).dropna(subset=["date"])
    df["security_id"] = df["tickers"].str.split(",")
    long = df.explode("security_id")[["date", "security_id"]]
    long["security_id"] = long["security_id"].map(normalize_ticker)
    long = long[long["security_id"].str.len() > 0]
    return long.drop_duplicates().sort_values(["date", "security_id"]).reset_index(drop=True)


def membership_from_snapshots(long: pd.DataFrame) -> pd.DataFrame:
    """Intervals [start, end) from composition snapshots taken at change dates.

    Names present in the first snapshot get ``start = NaT`` (member since before
    the dataset begins); names in the last snapshot get ``end = NaT``.
    """
    snapshots = long.groupby("date")["security_id"].apply(set).sort_index()
    intervals: list[tuple[str, pd.Timestamp | None, pd.Timestamp | None]] = []
    open_since: dict[str, pd.Timestamp | None] = {}
    previous: set[str] = set()
    for i, (date, members) in enumerate(snapshots.items()):
        for ticker in members - previous:
            open_since[ticker] = None if i == 0 else date
        for ticker in previous - members:
            intervals.append((ticker, open_since.pop(ticker), date))
        previous = members
    intervals.extend((t, start, None) for t, start in open_since.items())
    out = pd.DataFrame(intervals, columns=["security_id", "start", "end"])
    return out.astype({"start": "datetime64[ns]", "end": "datetime64[ns]"})


def reconcile_with_current(
    membership: pd.DataFrame, current: list[str], as_of: pd.Timestamp
) -> pd.DataFrame:
    """Bring the dataset up to date with today's constituents.

    The CSV is refreshed periodically, so it can lag a few weeks. Names that are
    members today but not open in the dataset start at ``as_of`` (the dataset's
    last date); open names no longer members end there.
    """
    m = membership.copy()
    current_set = set(current)
    open_rows = m["end"].isna()
    stale = open_rows & ~m["security_id"].isin(current_set)
    m.loc[stale, "end"] = as_of
    open_now = set(m.loc[m["end"].isna(), "security_id"])
    new = sorted(current_set - open_now)
    if new:
        m = pd.concat(
            [m, pd.DataFrame({"security_id": new, "start": as_of, "end": pd.NaT})],
            ignore_index=True,
        )
    return m.astype({"start": "datetime64[ns]", "end": "datetime64[ns]"})


def is_reused_ticker_tag(security_id: str) -> bool:
    return bool(REUSED_SUFFIX.search(security_id))
