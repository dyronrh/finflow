import numpy as np
import pandas as pd
import pytest

from quant_core.config import RiskConfig
from quant_core.risk.alerts import drawdown_alert, risk_alerts, signal_alerts
from quant_core.risk.enforce import enforce_risk_limits
from quant_core.risk.metrics import compute_risk, risk_contributions
from quant_core.risk.model import shrunk_covariance, trailing_returns


def _returns(n_days=300, vols=(0.10, 0.20, 0.30, 0.60), seed=0, corr=0.3):
    rng = np.random.default_rng(seed)
    k = len(vols)
    common = rng.normal(0, 1, n_days)
    idio = rng.normal(0, 1, (n_days, k))
    z = np.sqrt(corr) * common[:, None] + np.sqrt(1 - corr) * idio
    daily = z * (np.array(vols) / np.sqrt(252))
    names = [f"S{i}" for i in range(k)]
    return pd.DataFrame(daily, index=pd.bdate_range("2023-01-02", periods=n_days), columns=names)


def test_covariance_recovers_volatilities():
    rets = _returns(n_days=2000)
    cov = shrunk_covariance(rets)
    vols = np.sqrt(np.diag(cov))
    np.testing.assert_allclose(vols, [0.10, 0.20, 0.30, 0.60], rtol=0.25)
    assert np.all(np.linalg.eigvalsh(cov.to_numpy()) > 0)  # positive definite


def test_short_history_name_gets_median_variance_and_no_correlation():
    rets = _returns()
    rets.loc[rets.index[:250], "S3"] = np.nan  # recent listing
    cov = shrunk_covariance(rets)
    assert cov.loc["S3", "S0"] == 0.0
    assert cov.loc["S3", "S3"] == pytest.approx(
        np.median(np.diag(cov.loc[["S0", "S1", "S2"], ["S0", "S1", "S2"]])), rel=1e-6
    )


def test_risk_contributions_sum_to_one_and_track_volatility():
    rets = _returns(n_days=1000)
    cov = shrunk_covariance(rets)
    w = pd.Series(0.25, index=rets.columns)
    rc = risk_contributions(w, cov)
    assert rc.sum() == pytest.approx(1.0)
    assert rc.idxmax() == "S3" and rc.idxmin() == "S0"


def test_var_and_breaches():
    rets = _returns(n_days=500, vols=(0.6, 0.6, 0.6, 0.6), corr=0.8)
    w = pd.Series(0.25, index=rets.columns)
    limits = RiskConfig(max_portfolio_volatility=0.20, max_var_95_daily=0.02)
    report = compute_risk(w, rets, limits)
    assert report.volatility_annual > 0.4
    assert report.var_95_daily == pytest.approx(
        1.645 * report.volatility_annual / np.sqrt(252), rel=1e-3
    )
    assert report.cvar_95_daily > report.var_95_daily
    codes = {b.code for b in report.breaches}
    assert {"PORTFOLIO_VOLATILITY", "VAR_95"} <= codes
    assert all(a.severity == "CRITICAL" for a in risk_alerts(report) if a.code == "VAR_95")


def test_beta_against_benchmark():
    rets = _returns(n_days=800, corr=0.0)
    bench = rets.mean(axis=1)
    w = pd.Series(0.25, index=rets.columns)
    report = compute_risk(w, rets, RiskConfig(), benchmark_returns=bench)
    assert report.beta == pytest.approx(1.0, abs=1e-6)  # portfolio == benchmark


def test_enforcement_caps_industry_risk_and_exposure():
    rets = _returns(n_days=600, vols=(0.15, 0.15, 0.15, 0.15, 0.15, 0.9), corr=0.2)
    names = rets.columns
    w = pd.Series(1 / 6, index=names)
    sectors = pd.Series(["a", "b", "c", "d", "e", "f"], index=names)
    industries = pd.Series(["i1", "i1", "i2", "i3", "i4", "i5"], index=names)
    limits = RiskConfig(
        max_industry_weight=0.25,
        max_single_name_risk_contribution=0.30,
        max_portfolio_volatility=0.08,
        min_gross_exposure=0.7,
    )
    out, actions = enforce_risk_limits(w, rets, limits, sectors, industries, 0.5, 0.5)
    assert out.groupby(industries).sum().max() <= 0.25 + 1e-6
    rc = risk_contributions(out, shrunk_covariance(rets))
    assert rc.max() <= 0.30 + 1e-3
    assert out.sum() >= 0.7 - 1e-9  # never de-risks below the floor
    assert "RISK_CONTRIBUTION_TRIMMED" in actions
    assert any(a.startswith("EXPOSURE_SCALED") for a in actions)


def test_enforcement_is_noop_when_within_limits():
    rets = _returns(n_days=600, vols=(0.1,) * 10, corr=0.1)
    w = pd.Series(0.1, index=rets.columns)
    groups = pd.Series([f"g{i}" for i in range(10)], index=rets.columns)
    limits = RiskConfig(max_single_name_risk_contribution=0.2)
    out, actions = enforce_risk_limits(w, rets, limits, groups, groups, 0.2, 0.3)
    pd.testing.assert_series_equal(out.sort_index(), w.sort_index(), check_names=False)
    assert actions == []


def test_trailing_returns_are_point_in_time():
    close = pd.DataFrame(
        {"A": np.arange(1.0, 401.0)}, index=pd.bdate_range("2023-01-02", periods=400)
    )
    as_of = close.index[300]
    r = trailing_returns(close, as_of, ["A", "MISSING"], lookback_days=100)
    assert r.index.max() == as_of and len(r) == 100 and list(r.columns) == ["A"]


def test_signal_and_drawdown_alerts():
    prev = pd.DataFrame(
        {
            "security_id": ["A", "B", "C"],
            "decision": ["LONG", "LONG", "WATCH"],
            "composite_score": [80.0, 75.0, 70.0],
        }
    )
    cur = pd.DataFrame(
        {
            "security_id": ["A", "C", "D"],
            "decision": ["REDUCE", "STRONG_LONG", "STRONG_LONG"],
            "composite_score": [60.0, 90.0, 92.0],
        }
    )
    alerts = {(a.code, a.security_id) for a in signal_alerts(prev, cur, {"A", "B"})}
    assert ("HOLDING_DOWNGRADED", "A") in alerts
    assert ("SCORE_DROP", "A") in alerts
    assert ("HOLDING_LEFT_UNIVERSE", "B") in alerts
    assert ("NEW_STRONG_LONG", "C") in alerts and ("NEW_STRONG_LONG", "D") in alerts

    equity = pd.Series([100, 120, 95])
    alert = drawdown_alert(equity, threshold=0.10)
    assert alert is not None and alert.severity == "CRITICAL"  # -20.8%
    assert drawdown_alert(pd.Series([100, 101, 100]), 0.10) is None
