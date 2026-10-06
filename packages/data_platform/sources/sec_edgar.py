"""Point-in-time quarterly fundamentals from SEC EDGAR XBRL ``companyfacts``.

Every XBRL fact carries ``filed`` (the date the filing reached EDGAR), which is
used as ``available_at``. For each (concept, period) only the **first-filed**
value is kept, so later restatements never leak backwards in time.

Quarterly values are derived as follows:
* direct three-month facts (10-Q income statements);
* otherwise, differences of year-to-date facts that share a start date
  (6M − 3M, 9M − 6M, FY − 9M). This recovers Q4 from the 10-K and the
  quarterly cash-flow statement, which filers only report year-to-date.

Concept choices are approximations of standard definitions; financial
companies (banks, insurers) map poorly to EBITDA / gross margin and end up
with partial coverage, which the scoring layer handles.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEC_BASE = "https://data.sec.gov"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FORMS = ("10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT", "10-QT")

QUARTER_DAYS = (75, 105)
ANNUAL_DAYS = (340, 390)

CONCEPTS: dict[str, list[str]] = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "gross_profit": ["GrossProfit"],
    "cost_of_revenue": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
    ],
    "operating_income": ["OperatingIncomeLoss"],
    "depreciation": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
}
INSTANT_CONCEPTS: dict[str, list[str]] = {
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "Cash",
    ],
    "long_term_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "current_debt": ["DebtCurrent", "LongTermDebtCurrent"],
    "short_term_borrowings": ["ShortTermBorrowings", "CommercialPaper"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
}
SHARE_CONCEPTS = ("WeightedAverageNumberOfDilutedSharesOutstanding",)


# --------------------------------------------------------------------------- fetch
class SecClient:
    """Minimal EDGAR client with on-disk raw cache (bronze layer, never overwritten)."""

    def __init__(self, user_agent: str, cache_dir: Path, pause_seconds: float = 0.12) -> None:
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "SEC requires a User-Agent with contact info, e.g. "
                "SEC_USER_AGENT='finflow research you@example.com'"
            )
        import requests  # optional dependency

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip"})
        self._cache = Path(cache_dir)
        self._pause = pause_seconds

    def _get_json(self, url: str, cache_name: str) -> dict:
        path = self._cache / cache_name
        if path.exists():
            return json.loads(path.read_text())
        response = self._session.get(url, timeout=60)
        time.sleep(self._pause)  # SEC fair-access: max 10 requests/second
        if response.status_code == 404:
            return {}
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text)
        return response.json()

    def ticker_to_cik(self) -> dict[str, int]:
        data = self._get_json(TICKERS_URL, "company_tickers.json")
        return {
            row["ticker"].upper().replace(".", "-"): int(row["cik_str"]) for row in data.values()
        }

    def company_facts(self, cik: int) -> dict:
        return self._get_json(
            f"{SEC_BASE}/api/xbrl/companyfacts/CIK{cik:010d}.json",
            f"companyfacts/CIK{cik:010d}.json",
        )

    def submissions(self, cik: int) -> dict:
        return self._get_json(
            f"{SEC_BASE}/submissions/CIK{cik:010d}.json", f"submissions/CIK{cik:010d}.json"
        )


# --------------------------------------------------------------------------- parse
def _facts(company_facts: dict, concept: str, unit: str) -> pd.DataFrame:
    try:
        rows = company_facts["facts"]["us-gaap"][concept]["units"][unit]
    except KeyError:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if df.empty or "form" not in df.columns:
        return pd.DataFrame()
    df = df[df["form"].isin(FORMS)].copy()
    if df.empty:
        return df
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    if "start" in df.columns:
        df["start"] = pd.to_datetime(df["start"])
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    keys = ["start", "end"] if "start" in df.columns else ["end"]
    # First-filed value per period: restatements filed later are ignored.
    return df.sort_values("filed").drop_duplicates(keys, keep="first").reset_index(drop=True)


def quarterly_flow(company_facts: dict, concept: str, unit: str = "USD") -> pd.DataFrame:
    """Quarterly series (end, value, available_at) for a duration concept."""
    df = _facts(company_facts, concept, unit)
    if df.empty or "start" not in df.columns:
        return pd.DataFrame(columns=["end", "value", "available_at"])
    df["days"] = (df["end"] - df["start"]).dt.days

    out: dict[pd.Timestamp, tuple[float, pd.Timestamp]] = {}
    direct = df[df["days"].between(*QUARTER_DAYS)]
    for row in direct.itertuples():
        out[row.end] = (row.val, row.filed)

    # Year-to-date chains that share a fiscal-year start date.
    for _, chain in df[df["days"] <= ANNUAL_DAYS[1]].groupby("start"):
        chain = chain.sort_values("end")
        rows = list(chain.itertuples())
        for prev, cur in itertools.pairwise(rows):
            gap = (cur.end - prev.end).days
            if cur.end in out or not (QUARTER_DAYS[0] <= gap <= QUARTER_DAYS[1]):
                continue
            out[cur.end] = (cur.val - prev.val, max(cur.filed, prev.filed))

    if not out:
        return pd.DataFrame(columns=["end", "value", "available_at"])
    result = pd.DataFrame(
        [(end, v, f) for end, (v, f) in out.items()], columns=["end", "value", "available_at"]
    )
    return result.sort_values("end").reset_index(drop=True)


def instant_values(company_facts: dict, concept: str, unit: str = "USD") -> pd.DataFrame:
    df = _facts(company_facts, concept, unit)
    if df.empty:
        return pd.DataFrame(columns=["end", "value", "available_at"])
    if "start" in df.columns:
        df = df[df["start"].isna()]
    return df.rename(columns={"val": "value", "filed": "available_at"})[
        ["end", "value", "available_at"]
    ]


def _first_available(company_facts: dict, concepts: list[str], fn) -> pd.DataFrame:
    """Merge alternative concepts: earlier entries in ``concepts`` win per period."""
    merged = pd.DataFrame(columns=["end", "value", "available_at"])
    for concept in concepts:
        series = fn(company_facts, concept)
        if series.empty:
            continue
        new = series[~series["end"].isin(merged["end"])]
        merged = pd.concat([merged, new], ignore_index=True) if len(merged) else new
    return merged


def _attach(base: pd.DataFrame, series: pd.DataFrame, name: str) -> pd.DataFrame:
    """Join a series to quarter ends, tolerating ±7 days of calendar mismatch."""
    if series.empty:
        base[name] = np.nan
        base[f"{name}_available_at"] = pd.NaT
        return base
    right = series.rename(columns={"value": name, "available_at": f"{name}_available_at"})
    right = right.sort_values("end")
    return pd.merge_asof(
        base.sort_values("end"),
        right,
        on="end",
        direction="nearest",
        tolerance=pd.Timedelta(days=7),
    )


def parse_company_facts(company_facts: dict, security_id: str) -> pd.DataFrame:
    """Quarterly point-in-time fundamentals in the ``MarketData.fundamentals`` schema."""
    flows = {
        name: _first_available(company_facts, concepts, quarterly_flow)
        for name, concepts in CONCEPTS.items()
    }
    revenue = flows["revenue"]
    if revenue.empty:
        return pd.DataFrame()

    base = revenue[["end"]].drop_duplicates().copy()
    for name, series in flows.items():
        base = _attach(base, series, name)
    for name, concepts in INSTANT_CONCEPTS.items():
        base = _attach(base, _first_available(company_facts, concepts, instant_values), name)

    available_cols = [c for c in base.columns if c.endswith("_available_at")]
    # A row is usable only once every value in it has been filed.
    base["available_at"] = base[available_cols].max(axis=1)

    gross = base["gross_profit"].fillna(base["revenue"] - base["cost_of_revenue"])
    ebitda = base["operating_income"] + base["depreciation"]
    capex = base["capex"]
    if not any(c in company_facts.get("facts", {}).get("us-gaap", {}) for c in CONCEPTS["capex"]):
        capex = capex.fillna(0.0)  # filer never reports capex (e.g. some financials)
    debt = (
        base["long_term_debt"].fillna(0.0)
        + base["current_debt"].fillna(0.0)
        + base["short_term_borrowings"].fillna(0.0)
    )
    has_debt_data = (
        base[["long_term_debt", "current_debt", "short_term_borrowings"]].notna().any(axis=1)
    )
    debt = debt.where(has_debt_data)
    cash = base["cash"]

    out = pd.DataFrame(
        {
            "security_id": security_id,
            "event_time": base["end"],
            "available_at": base["available_at"],
            "revenue": base["revenue"],
            "gross_profit": gross,
            "ebitda": ebitda,
            "net_income": base["net_income"],
            "free_cash_flow": base["operating_cash_flow"] - capex,
            "invested_capital": base["equity"] + debt.fillna(0.0) - cash.fillna(0.0),
            "net_debt": debt - cash.fillna(0.0),
        }
    )
    out["ingested_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    return out.dropna(subset=["available_at"]).sort_values("event_time").reset_index(drop=True)


def parse_share_counts(company_facts: dict, security_id: str) -> pd.DataFrame:
    """Diluted weighted-average shares with their filing dates (not split-adjusted)."""
    frames = []
    for concept in SHARE_CONCEPTS:
        df = _facts(company_facts, concept, "shares")
        if df.empty or "start" not in df.columns:
            continue
        days = (df["end"] - df["start"]).dt.days
        df = df[days.between(QUARTER_DAYS[0], ANNUAL_DAYS[1])]
        frames.append(df[["end", "val", "filed"]])
    if not frames:
        try:
            rows = company_facts["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"][
                "shares"
            ]
        except KeyError:
            return pd.DataFrame(columns=["security_id", "available_at", "shares_outstanding"])
        df = pd.DataFrame(rows)
        df = df[df["form"].isin(FORMS)]
        df["end"] = pd.to_datetime(df["end"])
        df["filed"] = pd.to_datetime(df["filed"])
        frames.append(df[["end", "val", "filed"]])
    df = pd.concat(frames).sort_values("filed").drop_duplicates("end", keep="first")
    return pd.DataFrame(
        {
            "security_id": security_id,
            "available_at": df["filed"].to_numpy(),
            "shares_outstanding": pd.to_numeric(df["val"], errors="coerce").to_numpy(),
        }
    ).dropna()
