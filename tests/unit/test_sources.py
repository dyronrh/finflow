import pandas as pd
import pytest

from data_platform.sources import sec_edgar, sp500
from data_platform.sources.sic import sic_to_sector
from data_platform.sources.yahoo import split_factor_after, to_yahoo_symbol


def fact(start, end, val, filed, form="10-Q"):
    row = {"end": end, "val": val, "filed": filed, "form": form}
    if start:
        row["start"] = start
    return row


def company(us_gaap: dict, dei: dict | None = None) -> dict:
    facts = {"us-gaap": {k: {"units": u} for k, u in us_gaap.items()}}
    if dei:
        facts["dei"] = {k: {"units": u} for k, u in dei.items()}
    return {"facts": facts}


FY = [  # fiscal year 2023, calendar quarters
    fact("2023-01-01", "2023-03-31", 100, "2023-05-01"),
    fact("2023-04-01", "2023-06-30", 110, "2023-08-01"),
    fact("2023-07-01", "2023-09-30", 120, "2023-11-01"),
    fact("2023-01-01", "2023-12-31", 460, "2024-02-15", form="10-K"),
]


def test_q4_is_derived_from_annual_minus_ytd_and_dated_at_10k_filing():
    facts = company(
        {
            "Revenues": {"USD": FY},
        }
    )
    # The 10-Q for Q3 also reports the 9M YTD figure.
    facts["facts"]["us-gaap"]["Revenues"]["units"]["USD"].append(
        fact("2023-01-01", "2023-09-30", 330, "2023-11-01")
    )
    q = sec_edgar.quarterly_flow(facts, "Revenues").set_index("end")
    assert q.loc[pd.Timestamp("2023-12-31"), "value"] == 130
    assert q.loc[pd.Timestamp("2023-12-31"), "available_at"] == pd.Timestamp("2024-02-15")
    assert q.loc[pd.Timestamp("2023-03-31"), "available_at"] == pd.Timestamp("2023-05-01")


def test_cash_flow_quarters_come_from_ytd_differences():
    cfo = [
        fact("2023-01-01", "2023-03-31", 50, "2023-05-01"),
        fact("2023-01-01", "2023-06-30", 120, "2023-08-01"),
        fact("2023-01-01", "2023-09-30", 160, "2023-11-01"),
        fact("2023-01-01", "2023-12-31", 250, "2024-02-15", form="10-K"),
    ]
    q = sec_edgar.quarterly_flow(
        company({"NetCashProvidedByUsedInOperatingActivities": {"USD": cfo}}),
        "NetCashProvidedByUsedInOperatingActivities",
    )
    assert q["value"].tolist() == [50, 70, 40, 90]


def test_restatements_filed_later_are_ignored():
    rows = [
        fact("2023-01-01", "2023-03-31", 100, "2023-05-01"),
        fact("2023-01-01", "2023-03-31", 80, "2024-05-01"),  # restated a year later
    ]
    q = sec_edgar.quarterly_flow(company({"Revenues": {"USD": rows}}), "Revenues")
    assert q["value"].tolist() == [100]
    assert q["available_at"].tolist() == [pd.Timestamp("2023-05-01")]


def test_parse_company_facts_builds_fundamentals_rows():
    facts = company(
        {
            "Revenues": {"USD": FY},
            "GrossProfit": {
                "USD": [fact(r["start"], r["end"], r["val"] / 2, r["filed"], r["form"]) for r in FY]
            },
            "NetIncomeLoss": {
                "USD": [
                    fact(r["start"], r["end"], r["val"] / 10, r["filed"], r["form"]) for r in FY
                ]
            },
            "StockholdersEquity": {
                "USD": [
                    fact(None, "2023-03-31", 1000, "2023-05-01"),
                    fact(None, "2023-12-31", 1100, "2024-02-15", "10-K"),
                ]
            },
            "CashAndCashEquivalentsAtCarryingValue": {
                "USD": [fact(None, "2023-12-31", 300, "2024-02-20", "10-K")]
            },
        }
    )
    df = sec_edgar.parse_company_facts(facts, "XYZ")
    assert len(df) == 4
    q4 = df.set_index("event_time").loc[pd.Timestamp("2023-12-31")]
    assert q4["revenue"] == 130 and q4["gross_profit"] == 65
    # Row is available only when its latest component was filed.
    assert q4["available_at"] == pd.Timestamp("2024-02-20")
    assert (df["available_at"] > df["event_time"]).all()


def test_share_counts_keep_filing_dates():
    facts = company(
        {
            "WeightedAverageNumberOfDilutedSharesOutstanding": {
                "shares": [fact("2023-01-01", "2023-03-31", 1e9, "2023-05-01")]
            }
        }
    )
    s = sec_edgar.parse_share_counts(facts, "XYZ")
    assert s["available_at"].tolist() == [pd.Timestamp("2023-05-01")]


def test_membership_intervals_from_change_log():
    changes = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2022-06-01"]),
            "added": ["NEW1", "NEW2"],
            "removed": ["OLD1", "OLD2"],
        }
    )
    m = sp500.membership_intervals(["KEEP", "NEW1", "NEW2"], changes).set_index("security_id")
    assert m.loc["NEW2", "start"] == pd.Timestamp("2022-06-01") and pd.isna(m.loc["NEW2", "end"])
    assert m.loc["OLD2", "end"] == pd.Timestamp("2022-06-01")
    assert pd.isna(m.loc["OLD1", "start"]) and m.loc["OLD1", "end"] == pd.Timestamp("2020-01-01")
    assert pd.isna(m.loc["KEEP", "start"]) and pd.isna(m.loc["KEEP", "end"])


def test_parse_wikipedia_change_table_with_multiindex():
    table = pd.DataFrame(
        [["June 23, 2025", "ABC", "Abc Inc", "BRK.B", "Old Co", "reason"]],
        columns=pd.MultiIndex.from_tuples(
            [
                ("Effective Date", "Effective Date"),
                ("Added", "Ticker"),
                ("Added", "Security"),
                ("Removed", "Ticker"),
                ("Removed", "Security"),
                ("Reason", "Reason"),
            ]
        ),
    )
    out = sp500.parse_changes(table)
    assert out.iloc[0]["added"] == "ABC" and out.iloc[0]["removed"] == "BRK-B"
    assert out.iloc[0]["date"] == pd.Timestamp("2025-06-23")


@pytest.mark.parametrize(
    "sic,sector",
    [
        (7372, "information_technology"),
        (2834, "health_care"),
        (6022, "financials"),
        (1311, "energy"),
        (4911, "utilities"),
        (None, None),
    ],
)
def test_sic_mapping(sic, sector):
    assert sic_to_sector(sic) == sector


def test_yahoo_helpers():
    assert to_yahoo_symbol("brk.b") == "BRK-B"
    splits = pd.DataFrame(
        {
            "security_id": ["A", "A"],
            "date": pd.to_datetime(["2020-08-31", "2024-06-10"]),
            "ratio": [4.0, 10.0],
        }
    )
    assert split_factor_after(splits, "A", pd.Timestamp("2019-01-01")) == 40.0
    assert split_factor_after(splits, "A", pd.Timestamp("2021-01-01")) == 10.0
    assert split_factor_after(splits, "B", pd.Timestamp("2021-01-01")) == 1.0
