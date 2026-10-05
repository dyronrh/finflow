import numpy as np
import pandas as pd
import pytest

from backtesting.metrics import max_drawdown, performance_summary
from quant_core.execution.costs import TransactionCostModel


def test_costs_are_symmetric_and_non_negative():
    model = TransactionCostModel(commission_bps=1, half_spread_bps=2, slippage_bps=5)
    assert model.cost(10_000) == pytest.approx(8.0)
    assert model.cost(-10_000) == model.cost(10_000)
    assert sum(model.breakdown(10_000).values()) == pytest.approx(model.cost(10_000))


def test_max_drawdown():
    equity = pd.Series([100, 120, 60, 90, 130])
    assert max_drawdown(equity) == pytest.approx(-0.5)


def test_summary_on_constant_growth():
    equity = pd.Series(100 * np.cumprod(np.full(253, 1.0004)))
    s = performance_summary(equity)
    assert s["max_drawdown"] == 0
    assert s["cagr"] == pytest.approx(1.0004**252 - 1, rel=1e-6)
