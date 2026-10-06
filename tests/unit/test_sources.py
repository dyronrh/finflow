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


def _constituents():
    return pd.DataFrame(
        {
            "Symbol": ["AAA"],
            "Security": ["A"],
            "GICS Sector": ["Energy"],
            "GICS Sub-Industry": ["Oil"],
        }
    )


def test_select_tables_by_content_not_position():
    notice = pd.DataFrame({"Note": ["page banner"]})
    changes = pd.DataFrame(
        {"Date": ["2024-01-02"], "Added": ["AAA"], "Removed": ["ZZZ"], "Reason": ["x"]}
    )
    cons, chg = sp500.select_tables([notice, _constituents(), pd.DataFrame({"x": [1]}), changes])
    assert list(cons["Symbol"]) == ["AAA"]
    assert chg is changes


def test_select_tables_without_change_log():
    _, chg = sp500.select_tables([_constituents()])
    assert chg is None


def test_select_tables_raises_with_context_when_constituents_missing():
    with pytest.raises(ValueError, match="tables seen"):
        sp500.select_tables([pd.DataFrame({"x": [1]})])


def test_parse_changes_unnamed_date_column_and_footnotes():
    table = pd.DataFrame(
        [["July 22, 2025[5]", "ABC[a]", "Abc", "OLD", "Old", "r"], ["", "", "", "", "", ""]],
        columns=pd.MultiIndex.from_tuples(
            [
                ("Unnamed: 0_level_0", "Unnamed: 0_level_1"),
                ("Added", "Symbol"),
                ("Added", "Security"),
                ("Removed", "Symbol"),
                ("Removed", "Security"),
                ("Reason", "Reason"),
            ]
        ),
    )
    out = sp500.parse_changes(table)
    assert len(out) == 1
    assert out.iloc[0]["date"] == pd.Timestamp("2025-07-22")
    assert out.iloc[0]["added"] == "ABC" and out.iloc[0]["removed"] == "OLD"


def test_parse_changes_error_lists_columns():
    with pytest.raises(ValueError, match="not found in"):
        sp500.parse_changes(pd.DataFrame({"Date": ["2024-01-01"], "Foo": ["x"]}))


def test_membership_with_empty_change_log_keeps_current_members():
    empty = pd.DataFrame(columns=["date", "added", "removed"])
    m = sp500.membership_intervals(["A", "B"], empty)
    assert set(m["security_id"]) == {"A", "B"}
    assert m["start"].isna().all() and m["end"].isna().all()


# --------------------------------------------------------------- survivorship
from data_platform.real_market import _resolve_cik  # noqa: E402
from data_platform.sources import sp500_history  # noqa: E402


def test_history_snapshots_to_intervals():
    raw = pd.DataFrame(
        {
            "date": ["1996-01-02", "2010-05-03", "2015-01-05"],
            "tickers": ["AAA,BBB,BRK.B", "AAA,BRK.B,CCC", "BRK.B,CCC,DDD"],
        }
    )
    long = sp500_history.parse_components(raw)
    assert "BRK-B" in set(long["security_id"])
    m = sp500_history.membership_from_snapshots(long).set_index("security_id")
    assert pd.isna(m.loc["AAA", "start"]) and m.loc["AAA", "end"] == pd.Timestamp("2015-01-05")
    assert m.loc["BBB", "end"] == pd.Timestamp("2010-05-03")
    assert m.loc["CCC", "start"] == pd.Timestamp("2010-05-03") and pd.isna(m.loc["CCC", "end"])
    assert m.loc["DDD", "start"] == pd.Timestamp("2015-01-05")


def test_history_reconciled_with_current_constituents():
    m = pd.DataFrame(
        {"security_id": ["A", "B"], "start": [pd.NaT, pd.NaT], "end": [pd.NaT, pd.NaT]}
    ).astype({"start": "datetime64[ns]", "end": "datetime64[ns]"})
    out = sp500_history.reconcile_with_current(m, ["A", "C"], pd.Timestamp("2026-09-01"))
    out = out.set_index("security_id")
    assert out.loc["B", "end"] == pd.Timestamp("2026-09-01")  # left after dataset date
    assert out.loc["C", "start"] == pd.Timestamp("2026-09-01")  # joined after dataset date
    assert pd.isna(out.loc["A", "end"])


def test_reused_ticker_tags_are_detected():
    assert sp500_history.is_reused_ticker_tag("ABC-199912")
    assert not sp500_history.is_reused_ticker_tag("BRK-B")


def test_company_name_normalization_and_lookup():
    assert sec_edgar.normalize_company_name("The Walt Disney Co.") == "WALT DISNEY"
    assert sec_edgar.normalize_company_name("AT&T Inc.") == "AT AND T"
    lookup = sec_edgar.parse_cik_lookup(
        "WALT DISNEY CO:0001744489:\nWALT DISNEY CO /DE/:0001001039:\nXILINX INC:0000743988:\n"
    )
    assert lookup["XILINX"] == {743988}
    assert len(lookup["WALT DISNEY"]) == 2  # ambiguous: will not be used


def test_resolve_cik_does_not_trust_recycled_tickers():
    ticker_cik = {"OLD": 900}
    titles = {"OLD": "Brand New Company Inc"}
    names = {"OLD WIDGETS": {777}, "AMBIG": {1, 2}}
    # removed member, ticker now belongs to someone else → name lookup wins
    assert _resolve_cik("OLD", False, "Old Widgets Corp", {}, ticker_cik, titles, names) == (
        777,
        "sec_name_lookup",
    )
    # ticker owner's title matches the removed name → accepted
    titles_ok = {"OLD": "Old Widgets Inc"}
    assert _resolve_cik("OLD", False, "Old Widgets Corp", {}, ticker_cik, titles_ok, names)[1] == (
        "sec_ticker_name_checked"
    )
    # ambiguous name and no title match → unverified (caller checks filing overlap)
    assert _resolve_cik("OLD", False, "Ambig", {}, ticker_cik, titles, names)[1] == (
        "sec_ticker_unverified"
    )
    # current members use Wikipedia's CIK
    assert _resolve_cik("CUR", True, None, {"CUR": 5}, {}, {}, None) == (5, "wikipedia")
