import numpy as np
import pandas as pd

from quant_core.factors.normalization import winsorized_sector_percentile


def _frame(values, sectors):
    return pd.DataFrame({"x": values, "sector_id": sectors})


def test_percentiles_are_inside_open_unit_interval():
    rng = np.random.default_rng(0)
    df = _frame(rng.normal(size=200), rng.choice(list("ABCD"), 200))
    pct = winsorized_sector_percentile(df, "x")
    assert pct.between(0, 1, inclusive="neither").all()


def test_ranking_is_within_sector():
    df = _frame([1, 2, 3, 4, 5, 100, 200, 300, 400, 500], ["A"] * 5 + ["B"] * 5)
    pct = winsorized_sector_percentile(df, "x")
    np.testing.assert_allclose(pct[:5].to_numpy(), pct[5:].to_numpy())


def test_lower_is_better_flips_order():
    df = _frame([1, 2, 3, 4, 5], ["A"] * 5)
    pct = winsorized_sector_percentile(df, "x", higher_is_better=False)
    assert pct.iloc[0] > pct.iloc[-1]


def test_small_sector_falls_back_to_universe_rank():
    df = _frame([1, 2, 3, 4, 5, 6, 10], ["A"] * 6 + ["B"])
    pct = winsorized_sector_percentile(df, "x", min_group_size=5)
    assert pct.iloc[-1] == pct.max()  # lone name in B is ranked vs. the universe, not 0.5


def test_missing_values_stay_missing():
    df = _frame([1.0, np.nan, 3.0, 4.0, 5.0, 6.0], ["A"] * 6)
    pct = winsorized_sector_percentile(df, "x")
    assert np.isnan(pct.iloc[1])
    assert pct.notna().sum() == 5


def test_outliers_do_not_change_rank_order():
    df = _frame([1, 2, 3, 4, 1e9], ["A"] * 5)
    pct = winsorized_sector_percentile(df, "x", lower_quantile=0.0, upper_quantile=0.9)
    assert pct.is_monotonic_increasing
