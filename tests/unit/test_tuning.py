import numpy as np
import pandas as pd
import pytest

from backtesting.tuning import (
    apply_overrides,
    expand_grid,
    information_ratio,
    tune_strategy,
    walk_forward_selection,
)


def test_apply_overrides_sets_nested_fields(config):
    tuned = apply_overrides(
        config,
        {"portfolio.max_turnover_per_rebalance": 0.5, "signals.long_composite_percentile": 0.8},
    )
    assert tuned.portfolio.max_turnover_per_rebalance == 0.5
    assert tuned.signals.long_composite_percentile == 0.8
    assert config.portfolio.max_turnover_per_rebalance == 0.25  # base untouched


def test_apply_overrides_rejects_unknown_fields(config):
    with pytest.raises(KeyError):
        apply_overrides(config, {"portfolio.does_not_exist": 1})


def test_expand_grid():
    combos = expand_grid({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(combos) == 6 and {"a": 2, "b": "z"} in combos


def test_walk_forward_only_uses_past_to_choose():
    dates = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(0)
    bench = pd.Series(rng.normal(0, 0.01, 400), index=dates)
    good_then_bad = np.r_[np.full(200, 0.002), np.full(200, -0.002)]
    returns = pd.DataFrame(
        {
            "A": bench + good_then_bad + rng.normal(0, 0.001, 400),
            "B": bench + rng.normal(0, 0.001, 400),
        },
        index=dates,
    )
    oos, folds = walk_forward_selection(returns, bench, 100, 50, information_ratio)
    # Selection at each fold must be a function of the training window only:
    # A is chosen while its past was good, even though it later turns bad.
    first = folds.iloc[0]
    assert first["chosen_config"] == "A"
    assert first["train_end"] < first["test_start"]
    assert oos.index.is_monotonic_increasing and oos.index.is_unique
    assert oos.index[0] == dates[100]


def test_tune_strategy_end_to_end(market, config):
    report = tune_strategy(
        market,
        config,
        start="2023-01-01",
        end="2024-12-31",
        holdout_start="2024-07-01",
        grid={"portfolio.max_turnover_per_rebalance": [0.25, 1.0]},
        train_years=0.5,
        test_months=2,
    )
    assert set(report.grid["config_id"]) == {"base", "cfg_00", "cfg_01"}
    assert (report.folds["test_start"] < pd.Timestamp("2024-07-01")).all()
    assert report.oos_returns.index.max() < pd.Timestamp("2024-07-01")
    assert report.candidate.strategy_version.endswith("-tuned-rc")
    for block in (report.walk_forward, report.holdout):
        for stats in block.values():
            assert np.isfinite(stats["cagr"])
    assert {"composite", "value", "momentum"} <= set(report.ic_summary["factor"])
    assert "feature_earnings_yield" in set(report.ic_summary["factor"])
    assert "composite" in report.ic_yearly.columns


def test_weight_profiles_can_be_tuned(config):
    profile = {
        "value": 0.0,
        "growth": 0.25,
        "profitability": 0.3,
        "momentum": 0.3,
        "revisions": 0.15,
    }
    tuned = apply_overrides(config, {"composite_weights": profile})
    assert tuned.composite_weights == profile
