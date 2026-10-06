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


def fetch_tables(user_agent: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    import requests

    response = requests.get(WIKI_URL, headers={"User-Agent": user_agent}, timeout=60)
    response.raise_for_status()
    return select_tables(pd.read_html(StringIO(response.text)))


def _flat_columns(table: pd.DataFrame) -> list[str]:
    if isinstance(table.columns, pd.MultiIndex):
        return [" ".join(dict.fromkeys(str(p) for p in col)).strip() for col in table.columns]
    return [str(c).strip() for c in table.columns]


def select_tables(tables: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Find the constituents and change-log tables by their columns, not position.

    The page layout changes over time (extra tables, renamed headers), so the
    change log is the table whose headers mention both "added" and "removed".
    """
    constituents = changes = None
    for table in tables:
        cols = " | ".join(_flat_columns(table)).lower()
        if constituents is None and "symbol" in cols and "gics sector" in cols:
            constituents = table
        elif changes is None and "added" in cols and "removed" in cols:
            changes = table
    if constituents is None:
        found = [_flat_columns(t)[:6] for t in tables]
        raise ValueError(f"S&P 500 constituents table not found; tables seen: {found}")
    return constituents, changes


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
    df.columns = _flat_columns(df)
    cols = {c.lower(): c for c in df.columns}

    def pick(*options: tuple[str, ...]) -> str:
        for needles in options:
            for lower, original in cols.items():
                if all(n in lower for n in needles):
                    return original
        raise ValueError(f"change-log column {options} not found in {list(df.columns)}")

    date_col = (
        pick(("date",)) if any("date" in c for c in cols) else df.columns[0]
    )  # first column holds the effective date when the header is unnamed
    added_col = pick(("added", "ticker"), ("added", "symbol"))
    removed_col = pick(("removed", "ticker"), ("removed", "symbol"))

    def clean(value: object) -> str | None:
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return None
        text = re.sub(r"\[.*?\]", "", str(value)).strip().upper().replace(".", "-")
        return text or None

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(
                df[date_col].astype(str).str.replace(r"\[.*?\]", "", regex=True).str.strip(),
                errors="coerce",
                format="mixed",
            ),
            "added": df[added_col].map(clean),
            "removed": df[removed_col].map(clean),
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
