import pandas as pd
import pytest

from backtesting.engine import rebalance_dates, run_backtest


@pytest.fixture(scope="module")
def result(market, config):
    return run_backtest(market, config, "2023-01-01", "2024-12-31")


def test_no_same_bar_execution(result, market):
    trades = result.trades
    assert len(trades) > 0
    assert (trades["fill_date"] > trades["signal_date"]).all()
    for row in trades.sample(20, random_state=0).itertuples():
        assert row.fill_price == pytest.approx(market.open.loc[row.fill_date, row.security_id])


def test_costs_are_charged(result):
    assert (result.trades["cost_usd"] > 0).all()
    assert result.summary["total_costs_usd"] > 0


def test_holdings_respect_position_cap_at_fill(result, config):
    # Weights drift between rebalances; check right after each fill.
    fill_days = set(result.trades["fill_date"])
    at_fill = result.holdings[result.holdings["date"].isin(fill_days)]
    assert at_fill["weight"].max() <= config.portfolio.max_position_weight + 0.02


def test_backtest_is_deterministic(market, config, result):
    again = run_backtest(market, config, "2023-01-01", "2024-12-31")
    pd.testing.assert_series_equal(result.equity, again.equity)
    pd.testing.assert_frame_equal(result.trades, again.trades)


def test_benchmark_uses_same_calendar(result):
    assert result.equity.index.equals(result.benchmark_equity.index)


def test_metadata_is_recorded(result):
    for key in ("strategy_version", "data_version", "execution_assumption", "config_hash"):
        assert result.metadata[key]


def test_rebalance_dates_are_month_ends(market):
    dates = rebalance_dates(market.dates, "monthly")
    assert len(dates) == len({(d.year, d.month) for d in market.dates})
    biweekly = rebalance_dates(market.dates, "biweekly")
    assert len(biweekly) == 2 * len(dates)


def test_risk_limits_enforced_in_v020(market):
    from quant_core.config import load_strategy_config

    v1 = load_strategy_config("configs/strategies/v0.1.0.yaml")
    v2 = load_strategy_config("configs/strategies/v0.2.0.yaml")
    assert not v1.risk.enforce and v2.risk.enforce
    r1 = run_backtest(market, v1, "2023-06-01", "2024-12-31")
    r2 = run_backtest(market, v2, "2023-06-01", "2024-12-31")
    # v0.1.0 reports breaches but does not change weights; v0.2.0 removes them.
    assert (r1.rebalances["risk_actions"].fillna("") == "").all()
    assert r1.summary["rebalances_with_risk_breaches"] > 0
    assert r2.summary["rebalances_with_risk_breaches"] == 0
    assert {"ex_ante_volatility", "ex_ante_var_95", "ex_ante_beta"} <= set(r2.rebalances.columns)
