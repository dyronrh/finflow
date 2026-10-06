"""S&P 500 universe with approximate point-in-time membership.

Source: the Wikipedia constituents page (current members with GICS sector and
CIK, plus a dated table of additions and removals). Membership intervals are
rebuilt by walking the change log backwards from today.

Known limits (residual survivorship bias):
* removed companies often have no Yahoo price history any more and drop out;
* the change log is incomplete before ~2000 and tickers can be reused;
* GICS sectors are today's classification.
"""

from __future__ import annotations

import re
from io import StringIO

import pandas as pd

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def fetch_tables(user_agent: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    import requests

    html = requests.get(WIKI_URL, headers={"User-Agent": user_agent}, timeout=60).text
    tables = pd.read_html(StringIO(html))
    return tables[0], tables[1]


def snake(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def parse_constituents(table: pd.DataFrame) -> pd.DataFrame:
    df = table.rename(columns=lambda c: str(c).strip())
    out = pd.DataFrame(
        {
            "security_id": df["Symbol"].map(lambda s: str(s).strip().upper().replace(".", "-")),
            "ticker": df["Symbol"].astype(str).str.strip(),
            "sector_id": df["GICS Sector"].map(snake),
            "industry_id": df["GICS Sub-Industry"].map(snake),
        }
    )
    if "CIK" in df.columns:
        out["cik"] = pd.to_numeric(df["CIK"], errors="coerce").astype("Int64")
    return out


def parse_changes(table: pd.DataFrame) -> pd.DataFrame:
    """Normalise the change log to: date, added, removed (Yahoo-style tickers)."""
    df = table.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(dict.fromkeys(str(p) for p in col)).strip() for col in df.columns]
    cols = {c.lower(): c for c in df.columns}

    def pick(*needles: str) -> str:
        for lower, original in cols.items():
            if all(n in lower for n in needles):
                return original
        raise KeyError(needles)

    def clean(value: object) -> str | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = str(value).strip().upper().replace(".", "-")
        return text or None

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[pick("date")], errors="coerce"),
            "added": df[pick("added", "ticker")].map(clean),
            "removed": df[pick("removed", "ticker")].map(clean),
        }
    )
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def membership_intervals(current: list[str], changes: pd.DataFrame) -> pd.DataFrame:
    """Intervals [start, end) per ticker; NaT start means "before the change log"."""
    open_since: dict[str, pd.Timestamp | None] = {t: None for t in current}  # end of interval
    intervals: list[tuple[str, pd.Timestamp | None, pd.Timestamp | None]] = []
    members = set(current)
    for row in changes.sort_values("date", ascending=False).itertuples():
        if row.added and row.added in members:
            intervals.append((row.added, row.date, open_since.pop(row.added)))
            members.discard(row.added)
        if row.removed and row.removed not in members:
            members.add(row.removed)
            open_since[row.removed] = row.date
    intervals.extend((t, None, end) for t, end in open_since.items())
    return pd.DataFrame(intervals, columns=["security_id", "start", "end"]).astype(
        {"start": "datetime64[ns]", "end": "datetime64[ns]"}
    )
