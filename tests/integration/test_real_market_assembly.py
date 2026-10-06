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


def test_fetch_assemble_save_load_and_backtest(tmp_path, monkeypatch, config):
    tickers = [f"T{i:02d}" for i in range(30)]
    dates = pd.bdate_range("2018-01-01", "2024-12-31")
    rng = np.random.default_rng(1)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(dates), 30)), axis=0)),
        index=dates,
        columns=tickers,
    )
    close.iloc[:300, 0] = np.nan  # listed later

    sectors = ["information_technology", "health_care", "industrials"]
    constituents = pd.DataFrame(
        {
            "Symbol": tickers[:-2],
            "GICS Sector": [sectors[i % 3].replace("_", " ").title() for i in range(28)],
            "GICS Sub-Industry": ["Sub"] * 28,
            "CIK": range(1, 29),
        }
    )
    changes = pd.DataFrame(
        {
            "Date": ["January 2, 2020", "March 1, 2021"],
            "Added Ticker": ["T00", "T01"],
            "Removed Ticker": ["T28", "T29"],
        }
    )
    monkeypatch.setattr(real_market.sp500, "fetch_tables", lambda ua: (constituents, changes))
    monkeypatch.setattr(
        real_market.yahoo,
        "download_prices",
        lambda t, s, e: yahoo.YahooPrices(
            open=close.shift(1).fillna(close),
            close=close,
            volume=close * 0 + 1e6,
            splits=pd.DataFrame(
                {"security_id": ["T05"], "date": [pd.Timestamp("2022-06-01")], "ratio": [2.0]}
            ),
            failed=[],
        ),
    )
    monkeypatch.setattr(
        real_market.yahoo, "download_benchmark", lambda *a: close.mean(axis=1).rename("SPY")
    )

    class FakeSec:
        def __init__(self, *a, **k):
            pass

        def ticker_to_cik(self):
            return {"T28": 101, "T29": 102}

        def company_facts(self, cik):
            return _company_facts(1e7 * (1 + cik % 7))

        def submissions(self, cik):
            return {"sic": 7372}

    monkeypatch.setattr(real_market.sec_edgar, "SecClient", FakeSec)

    market = fetch_real_market("2018-01-01", "2024-12-31", "test agent test@example.com", tmp_path)
    assert len(market.securities) == 30
    assert (
        market.securities.set_index("security_id").loc["T29", "sector_id"]
        == "information_technology"
    )
    assert market.estimates.empty
    m = market.membership.set_index("security_id")
    assert m.loc["T01", "start"] == pd.Timestamp("2021-03-01")
    assert m.loc["T29", "end"] == pd.Timestamp("2021-03-01")
    # split after filing → share count expressed in post-split units
    s = market.shares
    early = s[(s.security_id == "T05") & (s.available_at < "2022-06-01")]["shares_outstanding"]
    assert (early == 2e8).all()

    reloaded = load_market(tmp_path / "processed")
    pd.testing.assert_frame_equal(reloaded.close, market.close, check_freq=False)
    assert reloaded.metadata["source"].startswith("yahoo")

    loose = config.model_copy(
        update={
            "eligibility": config.eligibility.model_copy(
                update={"min_market_cap_usd": 0, "min_adv_usd": 0}
            )
        }
    )
    result = run_backtest(reloaded, loose, "2021-01-01", "2024-12-31")
    assert np.isfinite(result.equity).all()
    assert len(result.trades) > 0
    assert "spy_cagr" in result.summary
    held = set(result.trades["security_id"])
    assert "T29" not in held or result.trades[result.trades.security_id == "T29"][
        "fill_date"
    ].min() < pd.Timestamp("2021-03-01")


def test_sec_parse_on_fixture_has_four_quarters_per_year():
    df = sec_edgar.parse_company_facts(_company_facts(1.0), "X")
    per_year = df.groupby(df["event_time"].dt.year).size()
    assert (per_year == 4).all()
    q4 = df[df["event_time"].dt.month == 12]
    assert (q4["revenue"] == 410 - 303).all()
