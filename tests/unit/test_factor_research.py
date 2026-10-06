from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from data_platform.features import _net_issuance, build_feature_snapshot
from research.factors import (
    CANDIDATES,
    Candidate,
    benjamini_hochberg,
    newey_west_t,
    research_factors,
)


def test_newey_west_without_lags_is_the_plain_t_stat():
    x = np.random.default_rng(0).normal(0.1, 1.0, 400)
    plain = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert newey_west_t(x, lags=0) == pytest.approx(plain)


def test_newey_west_shrinks_t_for_autocorrelated_series():
    rng = np.random.default_rng(1)
    shocks = rng.normal(0.05, 1.0, 600)
    overlapping = pd.Series(shocks).rolling(6).sum().dropna()  # like 6m returns sampled monthly
    assert abs(newey_west_t(overlapping, lags=6)) < abs(newey_west_t(overlapping, lags=0))


def test_benjamini_hochberg_matches_hand_computation():
    p = pd.Series({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5, "e": np.nan})
    q = benjamini_hochberg(p)
    # m = 4: sorted 0.01, 0.03, 0.04, 0.5 -> 0.04, 0.0533, 0.0533, 0.5
    assert q["a"] == pytest.approx(0.04)
    assert q["c"] == pytest.approx(0.04 * 4 / 3)
    assert q["b"] == pytest.approx(0.04 * 4 / 3)
    assert q["d"] == pytest.approx(0.5)
    assert np.isnan(q["e"])


def test_net_issuance_is_point_in_time(market):
    ids = list(market.securities["security_id"][:2])
    shares = pd.DataFrame(
        {
            "security_id": [ids[0], ids[0], ids[1], ids[1], ids[0]],
            "available_at": pd.to_datetime(
                ["2023-01-10", "2024-01-10", "2023-01-10", "2024-01-10", "2024-06-01"]
            ),
            "shares_outstanding": [100.0, 90.0, 100.0, 120.0, 50.0],
        }
    )
    m = dataclasses.replace(market, shares=shares)
    change = _net_issuance(m, pd.Timestamp("2024-02-01"), pd.Index(ids))
    assert change[ids[0]] == pytest.approx(-0.10)  # buyback; the June filing is not visible yet
    assert change[ids[1]] == pytest.approx(0.20)  # dilution


def test_new_candidate_features_are_computed(market, as_of):
    snap = build_feature_snapshot(market, as_of)
    for column in (
        "volatility_252d",
        "beta_252d",
        "high_52w_ratio",
        "sales_yield",
        "gross_profitability",
        "accruals",
        "earnings_volatility",
    ):
        assert snap[column].notna().mean() > 0.9, column
    assert (snap["high_52w_ratio"].dropna() <= 1.0 + 1e-12).all()
    # Synthetic stocks share one market factor: average beta close to 1.
    assert snap["beta_252d"].mean() == pytest.approx(1.0, abs=0.25)


def _with_planted_factors(market):
    """Wrap the feature provider with an 'oracle' that peeks one month ahead
    (must be found) and pure noise (must not)."""
    close = market.close
    ahead = close.shift(-21) / close - 1.0
    rng = np.random.default_rng(3)

    def provider(as_of: pd.Timestamp) -> pd.DataFrame:
        snap = build_feature_snapshot(market, as_of)
        future = ahead.loc[as_of].reindex(snap["security_id"]).to_numpy()
        snap["oracle"] = future + rng.normal(0, 0.02, len(snap))
        snap["noise"] = rng.normal(0, 1, len(snap))
        return snap

    return provider


def test_research_finds_a_planted_signal_and_rejects_noise(market, config):
    candidates = (
        Candidate("oracle", True, "test", "peeks at the future"),
        Candidate("noise", True, "test", "random"),
        *[c for c in CANDIDATES if c.name in ("mom_12_1", "roic")],
    )
    report = research_factors(
        market,
        config,
        start="2023-01-01",
        end="2024-12-31",
        holdout_start="2024-05-01",
        candidates=candidates,
        provider=_with_planted_factors(market),
    )
    v = report.verdicts
    assert v.loc["oracle", "verdict"] == "ACCEPT"
    assert v.loc["oracle", "passes_t3"]
    assert v.loc["noise", "verdict"] == "NOT_SIGNIFICANT"
    assert report.accepted[0] == "oracle"
    assert report.research.loc["oracle", "monotonicity"] > 0.8
    assert report.composite["holdout"]["mean_ic_1m"] > 0.2
    assert set(report.correlations.columns) >= {"oracle", "noise"}


def test_a_signal_against_the_hypothesis_is_reported_not_flipped(market, config):
    report = research_factors(
        market,
        config,
        start="2023-01-01",
        end="2024-12-31",
        holdout_start="2024-05-01",
        candidates=(Candidate("oracle", False, "test", "wrong sign on purpose"),),
        provider=_with_planted_factors(market),
    )
    assert report.verdicts.loc["oracle", "verdict"] == "INVERTED"
    assert report.accepted == []
