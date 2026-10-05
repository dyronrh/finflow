import numpy as np
import pandas as pd
import pytest

from data_platform.features import build_feature_snapshot
from quant_core.portfolio.construction import apply_weight_caps, build_target_weights
from quant_core.portfolio.rebalance import plan_rebalance
from quant_core.signals.rules import generate_signals


@pytest.fixture(scope="module")
def signals(market, config, as_of):
    return generate_signals(build_feature_snapshot(market, as_of), config)


def test_target_weights_respect_limits(signals, config):
    pcfg = config.portfolio
    w = build_target_weights(signals, pcfg)
    sectors = signals.set_index("security_id")["sector_id"].reindex(w.index)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()
    assert w.max() <= pcfg.max_position_weight + 1e-9
    assert w.groupby(sectors).sum().max() <= pcfg.max_sector_weight + 1e-9
    buyable = (~signals["decision"].isin(["REDUCE", "AVOID"])).sum()
    assert min(pcfg.min_holdings, buyable) <= len(w) <= pcfg.max_holdings
    assert (
        not signals.set_index("security_id")
        .loc[w.index, "decision"]
        .isin(["REDUCE", "AVOID"])
        .any()
    )


@pytest.mark.parametrize("seed", range(20))
def test_caps_hold_for_random_inputs(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(15, 40))
    w = pd.Series(rng.lognormal(size=n), index=[f"S{i}" for i in range(n)])
    sectors = pd.Series(rng.choice(list("ABCDEF"), n), index=w.index)
    capped = apply_weight_caps(w, sectors, max_position=0.08, max_sector=0.25)
    assert capped.max() <= 0.08 + 1e-9
    assert capped.groupby(sectors).sum().max() <= 0.25 + 1e-9
    assert capped.sum() <= 1.0 + 1e-9
    feasible = min(n * 0.08, sectors.nunique() * 0.25) >= 1.0
    if feasible:
        assert capped.sum() == pytest.approx(1.0, abs=1e-6)


def test_infeasible_caps_leave_cash():
    w = pd.Series([1.0, 1.0], index=["A", "B"])
    capped = apply_weight_caps(w, pd.Series(["X", "X"], index=w.index), 0.5, 0.25)
    assert capped.sum() == pytest.approx(0.25)


def test_trade_band_skips_small_changes():
    current = pd.Series({"A": 0.50, "B": 0.50})
    target = pd.Series({"A": 0.502, "B": 0.498})
    plan = plan_rebalance(current, target, trade_band=0.005, max_turnover=1.0)
    assert plan.turnover == 0.0
    assert set(plan.orders["side"]) == {"HOLD"}


def test_exits_trade_even_inside_band():
    current = pd.Series({"A": 0.003, "B": 0.997})
    target = pd.Series({"B": 1.0})
    plan = plan_rebalance(current, target, trade_band=0.005, max_turnover=1.0)
    assert plan.orders.set_index("security_id").loc["A", "side"] == "SELL"


def test_turnover_cap_scales_trades():
    current = pd.Series({"A": 0.5, "B": 0.5})
    target = pd.Series({"C": 0.5, "D": 0.5})
    plan = plan_rebalance(current, target, trade_band=0.0, max_turnover=0.25)
    assert plan.turnover_capped
    assert plan.orders["delta_weight"].abs().sum() == pytest.approx(0.25)
    assert plan.final_weights.sum() == pytest.approx(1.0)


def test_initial_funding_is_exempt_from_turnover_cap():
    plan = plan_rebalance(pd.Series(dtype=float), pd.Series({"A": 0.6, "B": 0.4}), 0.0, 0.25)
    assert not plan.turnover_capped
    assert plan.final_weights.sum() == pytest.approx(1.0)
