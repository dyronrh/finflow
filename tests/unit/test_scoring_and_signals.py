import pandas as pd
import pytest

from data_platform.features import build_feature_snapshot
from data_platform.validation import validate_factor_scores
from quant_core.config import FACTOR_FAMILIES, StrategyConfig
from quant_core.factors.scoring import compute_factor_scores
from quant_core.signals.rules import Decision, generate_signals


@pytest.fixture(scope="module")
def signals(market, config, as_of):
    return generate_signals(build_feature_snapshot(market, as_of), config)


def test_scores_are_bounded(signals):
    for family in FACTOR_FAMILIES:
        assert signals[f"{family}_score"].dropna().between(0, 100).all()
        assert signals[f"{family}_percentile"].dropna().between(0, 1).all()
    assert signals["composite_score"].between(0, 100).all()


def test_scores_satisfy_data_contract(signals, as_of):
    result = validate_factor_scores(signals, as_of)
    assert result.ok, result.violations


def test_only_eligible_names_are_ranked(signals, config):
    assert (signals["price"] >= config.eligibility.minimum_stock_price_usd).all()
    assert (signals["adv_usd"] >= config.eligibility.min_adv_usd).all()
    assert (signals["market_cap_usd"] >= config.eligibility.min_market_cap_usd).all()


def test_decisions_follow_rules(signals, config):
    t = config.signals
    longs = signals[signals["decision"].isin([Decision.LONG, Decision.STRONG_LONG])]
    assert (longs["composite_percentile"] >= t.long_composite_percentile).all()
    assert (longs["profitability_percentile"] >= t.long_min_profitability_percentile).all()
    assert (longs["risk_flag"] == 0).all()
    strong = signals[signals["decision"] == Decision.STRONG_LONG]
    assert (strong["composite_percentile"] >= t.strong_long_composite_percentile).all()
    assert (signals.loc[signals["risk_flag"] == 1, "decision"] == Decision.AVOID).all()
    assert set(signals["decision"]) <= {d.value for d in Decision}


def test_every_signal_has_an_explanation(signals):
    assert signals["explanation"].map(len).gt(0).all()


def test_composite_uses_configured_weights():
    df = pd.DataFrame(
        {
            "security_id": [f"S{i}" for i in range(10)],
            "sector_id": ["A"] * 10,
            "earnings_yield": range(10),
            "revenue_growth": range(10),
            "roic": range(10),
            "mom_12_1": range(10),
            "eps_rev_30d": range(10),
        }
    )
    config = StrategyConfig()
    scored = compute_factor_scores(df, config)
    expected = sum(scored[f"{f}_score"] * w for f, w in config.composite_weights.items())
    pd.testing.assert_series_equal(scored["composite_score"], expected, check_names=False)


def test_low_coverage_names_get_no_composite():
    df = pd.DataFrame({"security_id": ["A", "B"], "sector_id": ["X", "X"], "mom_12_1": [0.1, 0.2]})
    scored = compute_factor_scores(df, StrategyConfig())
    assert scored["composite_score"].isna().all()


def test_invalid_weights_rejected():
    with pytest.raises(ValueError):
        StrategyConfig(
            composite_weights={
                "value": 0.5,
                "growth": 0.5,
                "profitability": 0.5,
                "momentum": 0.0,
                "revisions": 0.0,
            }
        )
