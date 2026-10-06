"""fetch_real_market wiring with all network sources stubbed out."""

import numpy as np
import pandas as pd

from backtesting.engine import run_backtest
from data_platform import real_market
from data_platform.real_market import fetch_real_market, load_market
from data_platform.sources import sec_edgar, yahoo


def _company_facts(scale: float) -> dict:
    rows, shares = [], []
    for year in range(2018, 2025):
        for q, (s, e, f) in enumerate(
            [("01-01", "03-31", "05-01"), ("04-01", "06-30", "08-01"), ("07-01", "09-30", "11-01")]
        ):
            rows.append(
                {
                    "start": f"{year}-{s}",
                    "end": f"{year}-{e}",
                    "val": scale * (100 + q),
                    "filed": f"{year}-{f}",
                    "form": "10-Q",
                }
            )
        rows.append(
            {
                "start": f"{year}-01-01",
                "end": f"{year}-09-30",
                "val": scale * 303,
                "filed": f"{year}-11-01",
                "form": "10-Q",
            }
        )
        rows.append(
            {
                "start": f"{year}-01-01",
                "end": f"{year}-12-31",
                "val": scale * 410,
                "filed": f"{year + 1}-02-15",
                "form": "10-K",
            }
        )
        shares.append(
            {
                "start": f"{year}-01-01",
                "end": f"{year}-12-31",
                "val": 1e8,
                "filed": f"{year + 1}-02-15",
                "form": "10-K",
            }
        )

    def scaled(k):
        return [{**r, "val": r["val"] * k} for r in rows]

    gaap = {
        "Revenues": {"units": {"USD": rows}},
        "GrossProfit": {"units": {"USD": scaled(0.4)}},
        "OperatingIncomeLoss": {"units": {"USD": scaled(0.2)}},
        "DepreciationDepletionAndAmortization": {"units": {"USD": scaled(0.05)}},
        "NetIncomeLoss": {"units": {"USD": scaled(0.1)}},
        "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": scaled(0.15)}},
        "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": scaled(0.03)}},
        "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": shares}},
    }
    return {"facts": {"us-gaap": gaap}}


def _stub_sources(monkeypatch, history_ok=True):
    """Universe: T00..T27 are current members. Former members:
    T28/T29 (removed 2020/2021), OLD1 (acquired mid-2022, prices stop),
    RECY (removed 2019; ticker now used by a company listed in 2023)."""
    current = [f"T{i:02d}" for i in range(28)]
    former = ["T28", "T29", "OLD1", "RECY"]
    dates = pd.bdate_range("2018-01-01", "2024-12-31")
    rng = np.random.default_rng(1)
    names = current + former
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(dates), len(names))), axis=0)),
        index=dates,
        columns=names,
    )
    close.iloc[:300, 0] = np.nan  # listed later
    close.loc[close.index > "2022-06-30", "OLD1"] = np.nan  # acquired
    close.loc[close.index < "2023-01-02", "RECY"] = np.nan  # recycled ticker

    sectors = ["information_technology", "health_care", "industrials"]
    constituents = pd.DataFrame(
        {
            "Symbol": current,
            "GICS Sector": [sectors[i % 3].replace("_", " ").title() for i in range(28)],
            "GICS Sub-Industry": ["Sub"] * 28,
            "CIK": range(1, 29),
        }
    )
    changes = pd.DataFrame(
        {
            "Date": ["June 1, 2019", "January 2, 2020", "March 1, 2021", "July 1, 2022"],
            "Added Ticker": ["T27", "T00", "T01", "T26"],
            "Removed Ticker": ["RECY", "T28", "T29", "OLD1"],
            "Removed Security": ["Recy Old Co", "T28 Inc", "T29 Inc", "Old One Corp"],
        }
    )
    snapshots = pd.DataFrame(
        {
            "date": ["2017-01-03", "2019-06-01", "2020-01-02", "2021-03-01", "2022-07-01"],
            "tickers": [
                ",".join([*current[2:26], "T28", "T29", "OLD1", "RECY"]),
                ",".join([*current[2:28], "T28", "T29", "OLD1"]),
                ",".join(["T00", *current[2:28], "T29", "OLD1"]),
                ",".join(["T00", "T01", *current[2:28], "OLD1"]),
                ",".join(current),
            ],
        }
    )
    snapshots.loc[0, "tickers"] = snapshots.loc[0, "tickers"].replace("T27,", "")
    monkeypatch.setattr(real_market.sp500, "fetch_tables", lambda ua: (constituents, changes))
    if history_ok:
        monkeypatch.setattr(real_market.sp500_history, "fetch_components_csv", lambda ua: snapshots)
    else:

        def boom(ua):
            raise OSError("network down")

        monkeypatch.setattr(real_market.sp500_history, "fetch_components_csv", boom)
    monkeypatch.setattr(
        real_market.yahoo,
        "download_prices",
        lambda t, s, e, **kw: yahoo.YahooPrices(
            open=close.shift(1).fillna(close)[[c for c in close.columns if c in t]],
            close=close[[c for c in close.columns if c in t]],
            volume=close[[c for c in close.columns if c in t]] * 0 + 1e6,
            splits=pd.DataFrame(
                {"security_id": ["T05"], "date": [pd.Timestamp("2022-06-01")], "ratio": [2.0]}
            ),
            failed=[],
            price_close=close[[c for c in close.columns if c in t]],
        ),
    )
    bench = {"SPY": close.mean(axis=1), "RSP": close.mean(axis=1) * 0.99}
    monkeypatch.setattr(
        real_market.yahoo, "download_benchmark", lambda sym, *a, **kw: bench[sym].rename(sym)
    )

    class FakeSec:
        def __init__(self, *a, **k):
            pass

        def ticker_to_cik(self):
            # OLD1 and RECY now belong to *other* companies.
            return {"T28": 101, "T29": 102, "OLD1": 900, "RECY": 901}

        def ticker_titles(self):
            return {
                "T28": "T28 Inc",
                "T29": "T29 Inc",
                "OLD1": "Totally Different Inc",
                "RECY": "New Co",
            }

        def name_to_ciks(self):
            return {"OLD ONE": {777}}

        def company_facts(self, cik):
            if cik == 900:  # the unrelated company: filings only since 2023
                facts = _company_facts(1e7)
                for rows in facts["facts"]["us-gaap"].values():
                    for unit in rows["units"].values():
                        unit[:] = [r for r in unit if r["end"] >= "2023"]
                return facts
            return _company_facts(1e7 * (1 + cik % 7))

        def submissions(self, cik):
            return {"sic": 7372}

    monkeypatch.setattr(real_market.sec_edgar, "SecClient", FakeSec)
    return close


def test_fetch_assemble_save_load_and_backtest(tmp_path, monkeypatch, config):
    _stub_sources(monkeypatch)
    market = fetch_real_market("2018-01-01", "2024-12-31", "test agent test@example.com", tmp_path)
    meta = market.metadata
    assert meta["universe"].startswith("point-in-time membership from fja05680")

    ids = set(market.securities["security_id"])
    # Former members with data are kept; the recycled ticker is not.
    assert {"T28", "T29", "OLD1"} <= ids
    assert "RECY" not in ids and meta["tickers_dropped_as_recycled"] == ["RECY"]
    # OLD1's ticker now maps to another company; the name lookup finds the right one.
    assert meta["cik_resolution"].get("sec_name_lookup") == 1
    assert (
        market.securities.set_index("security_id").loc["T29", "sector_id"]
        == "information_technology"
    )

    m = market.membership
    t01 = m[m.security_id == "T01"].iloc[0]
    assert t01["start"] == pd.Timestamp("2021-03-01")
    old1 = m[m.security_id == "OLD1"].iloc[0]
    assert old1["end"] == pd.Timestamp("2022-07-01")
    # Future members are not in the universe before they joined.
    assert "T01" not in market.members_at(pd.Timestamp("2020-06-30"))
    assert "OLD1" in market.members_at(pd.Timestamp("2020-06-30"))
    assert meta["coverage_by_year"][0]["with_prices"] > 0.9

    s = market.shares
    early = s[(s.security_id == "T05") & (s.available_at < "2022-06-01")]["shares_outstanding"]
    assert (early == 2e8).all()

    reloaded = load_market(tmp_path / "processed")
    pd.testing.assert_frame_equal(reloaded.close, market.close, check_freq=False)
    assert reloaded.equal_weight_reference is not None
    assert reloaded.metadata["warnings"] == []

    loose = config.model_copy(
        update={
            "eligibility": config.eligibility.model_copy(
                update={"min_market_cap_usd": 0, "min_adv_usd": 0}
            )
        }
    )
    result = run_backtest(reloaded, loose, "2019-01-01", "2024-12-31")
    assert np.isfinite(result.equity).all()
    assert {"spy_cagr", "rsp_cagr", "estimated_survivorship_bias_cagr"} <= set(result.summary)
    trades = result.trades
    # Never bought before joining the index / after leaving it.
    t01_buys = trades[(trades.security_id == "T01") & (trades.side == "BUY")]
    assert (t01_buys["signal_date"] >= pd.Timestamp("2021-03-01")).all()
    old1_buys = trades[(trades.security_id == "OLD1") & (trades.side == "BUY")]
    assert (old1_buys["signal_date"] < pd.Timestamp("2022-07-01")).all()


def test_falls_back_to_wikipedia_change_log(tmp_path, monkeypatch):
    _stub_sources(monkeypatch, history_ok=False)
    market = fetch_real_market("2018-01-01", "2024-12-31", "test agent test@example.com", tmp_path)
    assert market.metadata["universe"].startswith("approximate point-in-time")
    assert market.membership.set_index("security_id").loc["T01", "start"] == pd.Timestamp(
        "2021-03-01"
    )


def test_sec_parse_on_fixture_has_four_quarters_per_year():
    df = sec_edgar.parse_company_facts(_company_facts(1.0), "X")
    per_year = df.groupby(df["event_time"].dt.year).size()
    assert (per_year == 4).all()
    q4 = df[df["event_time"].dt.month == 12]
    assert (q4["revenue"] == 410 - 303).all()
