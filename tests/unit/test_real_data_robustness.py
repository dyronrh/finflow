"""Behaviour with real-data imperfections: gaps, missing factors, membership."""

import numpy as np
import pandas as pd

from backtesting.engine import run_backtest
from data_platform.features import build_feature_snapshot
from data_platform.market import MarketData
from quant_core.signals.rules import _decide, apply_eligibility, generate_signals


def _copy(market, **changes):
    fields = dict(
        securities=market.securities,
        open=market.open.copy(),
        close=market.close.copy(),
        volume=market.volume.copy(),
        fundamentals=market.fundamentals,
        estimates=market.estimates,
        seed=market.seed,
        data_version=market.data_version,
    )
    fields.update(changes)
    return MarketData(**fields)


def test_missing_market_cap_fails_eligibility(market, config, as_of):
    feats = build_feature_snapshot(market, as_of)
    feats.loc[0, "market_cap_usd"] = np.nan
    screened = apply_eligibility(feats, config)
    assert not screened.loc[0, "eligible"]


def test_without_estimates_scores_are_still_computed(market, config, as_of):
    no_estimates = _copy(market, estimates=pd.DataFrame())
    signals = generate_signals(build_feature_snapshot(no_estimates, as_of), config)
    assert signals["revisions_score"].isna().all()
    assert signals["composite_score"].notna().all()


def test_missing_factor_family_does_not_block_longs(config):
    frame = pd.DataFrame(
        {
            "composite_percentile": [0.97, 0.90],
            "profitability_percentile": [0.9, 0.9],
            "momentum_percentile": [0.9, 0.9],
            "revisions_percentile": [np.nan, np.nan],
            "liquidity_percentile": [0.9, 0.9],
            "risk_flag": [0, 0],
        }
    )
    assert _decide(frame, config).tolist() == ["STRONG_LONG", "LONG"]
    frame.loc[1, "revisions_percentile"] = 0.1  # data exists for one name → gate applies
    frame.loc[0, "revisions_percentile"] = 0.9
    assert _decide(frame, config).tolist() == ["STRONG_LONG", "REDUCE"]


def test_membership_restricts_universe(market, config, as_of):
    ids = market.securities["security_id"]
    membership = pd.DataFrame({"security_id": ids[:40], "start": pd.NaT, "end": pd.NaT}).astype(
        {"start": "datetime64[ns]", "end": "datetime64[ns]"}
    )
    restricted = _copy(market, membership=membership)
    signals = generate_signals(build_feature_snapshot(restricted, as_of), config)
    assert set(signals["security_id"]) <= set(ids[:40])


def test_backtest_survives_price_gaps(market, config):
    gappy = _copy(market)
    halted = gappy.securities["security_id"].iloc[:30]
    window = gappy.open.index[
        (gappy.open.index >= "2024-03-01") & (gappy.open.index < "2024-05-01")
    ]
    gappy.open.loc[window, halted] = np.nan
    gappy.close.loc[window, halted] = np.nan
    result = run_backtest(gappy, config, "2023-06-01", "2024-12-31")
    assert np.isfinite(result.equity).all()
    if len(result.unfilled):
        assert (result.unfilled["security_id"].isin(halted)).all()
    assert (result.trades["fill_price"] > 0).all()


def test_spy_benchmark_is_reported(market, config):
    with_spy = _copy(market, benchmark_close=market.close.mean(axis=1))
    result = run_backtest(with_spy, config, "2024-01-01", "2024-12-31")
    assert "spy_cagr" in result.summary and np.isfinite(result.summary["beta_vs_spy"])


def test_levels_use_traded_price_not_dividend_adjusted(market, as_of):
    """Market cap / ADV / price filter must not use dividend-adjusted levels
    (those embed future dividends); returns still use the adjusted series."""
    traded = market.close.copy()
    adjusted = market.close * 0.6  # as if 40% of the level were future dividends
    m = _copy(market, close=adjusted, open=market.open * 0.6, price_close=traded)
    snap = build_feature_snapshot(m, as_of).set_index("security_id")
    base = build_feature_snapshot(market, as_of).set_index("security_id")
    pd.testing.assert_series_equal(snap["market_cap_usd"], base["market_cap_usd"])
    pd.testing.assert_series_equal(snap["price"], base["price"])
    pd.testing.assert_series_equal(snap["earnings_yield"], base["earnings_yield"])
    # Constant scaling leaves returns-based features unchanged.
    np.testing.assert_allclose(snap["mom_12_1"], base["mom_12_1"])


def test_delisted_holdings_are_liquidated(market, config):
    """A held name whose quotes stop for good is sold at its last price."""
    first = run_backtest(market, config, "2023-06-01", "2024-12-31")
    held = first.holdings[first.holdings["date"] == pd.Timestamp("2024-03-28")]
    victim = held.sort_values("weight").iloc[-1]["security_id"]

    delisted = _copy(market)
    gone = delisted.close.index > "2024-04-15"
    for frame in (delisted.close, delisted.open, delisted.volume):
        frame.loc[gone, victim] = np.nan
    result = run_backtest(delisted, config, "2023-06-01", "2024-12-31")
    sells = result.trades[result.trades["side"] == "DELISTED_SELL"]
    assert victim in set(sells["security_id"])
    later = result.holdings[result.holdings["date"] > pd.Timestamp("2024-05-01")]
    assert victim not in set(later["security_id"])
    assert np.isfinite(result.equity).all()
    assert result.summary["delisted_liquidations"] >= 1


def test_no_signals_keeps_portfolio_instead_of_liquidating(market, config):
    calls = {"n": 0}

    def flaky(as_of, cfg):
        calls["n"] += 1
        signals = generate_signals(build_feature_snapshot(market, as_of), cfg)
        return signals.iloc[0:0] if calls["n"] == 3 else signals  # 3rd rebalance: data gap

    result = run_backtest(market, config, "2024-01-01", "2024-12-31", signal_provider=flaky)
    skipped = result.rebalances[result.rebalances.get("skipped").notna()]
    assert len(skipped) == 1
    day_after = result.holdings[result.holdings["date"] > skipped.iloc[0]["signal_date"]]
    assert len(day_after) > 0  # still invested
