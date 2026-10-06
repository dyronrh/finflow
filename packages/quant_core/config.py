"""Versioned strategy configuration.

Every parameter that can change a recommendation lives here so that a strategy
version fully determines the decision logic (README §8.3, §10.3).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

FactorFamily = Literal["value", "growth", "profitability", "momentum", "revisions"]

FACTOR_FAMILIES: tuple[FactorFamily, ...] = (
    "value",
    "growth",
    "profitability",
    "momentum",
    "revisions",
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class NormalizationConfig(_Frozen):
    lower_quantile: float = Field(0.01, ge=0.0, lt=0.5)
    upper_quantile: float = Field(0.99, gt=0.5, le=1.0)
    min_group_size: int = Field(5, ge=1)
    min_family_coverage: float = Field(0.5, gt=0.0, le=1.0)
    min_composite_coverage: float = Field(0.7, gt=0.0, le=1.0)


class EligibilityConfig(_Frozen):
    minimum_stock_price_usd: float = 5.0
    min_market_cap_usd: float = 2_000_000_000
    min_adv_usd: float = 5_000_000
    max_volatility_percentile: float = Field(0.95, gt=0.0, le=1.0)


class SignalThresholds(_Frozen):
    strong_long_composite_percentile: float = 0.95
    long_composite_percentile: float = 0.85
    long_min_profitability_percentile: float = 0.60
    long_min_momentum_percentile: float = 0.50
    long_min_revisions_percentile: float = 0.50
    long_min_liquidity_percentile: float = 0.60
    watch_composite_percentile: float = 0.70
    reduce_composite_percentile: float = 0.55
    reduce_revisions_percentile: float = 0.30
    avoid_composite_percentile: float = 0.20


class PortfolioConfig(_Frozen):
    weighting: Literal["equal_weight", "inverse_volatility"] = "inverse_volatility"
    max_position_weight: float = Field(0.08, gt=0.0, le=1.0)
    max_sector_weight: float = Field(0.25, gt=0.0, le=1.0)
    min_holdings: int = Field(20, ge=1)
    max_holdings: int = Field(35, ge=1)
    max_turnover_per_rebalance: float = Field(0.25, gt=0.0)
    trade_band: float = Field(0.005, ge=0.0)
    rebalance_frequency: Literal["monthly", "biweekly"] = "monthly"

    @model_validator(mode="after")
    def _check_holdings(self) -> PortfolioConfig:
        if self.min_holdings > self.max_holdings:
            raise ValueError("min_holdings must be <= max_holdings")
        return self


class RiskConfig(_Frozen):
    """Portfolio risk limits (README §10.3, §15). ``enforce`` makes them active
    constraints on target weights; otherwise they only raise alerts."""

    enforce: bool = False
    lookback_days: int = Field(252, ge=60)
    max_industry_weight: float = Field(0.15, gt=0.0, le=1.0)
    max_single_name_risk_contribution: float = Field(0.10, gt=0.0, le=1.0)
    max_portfolio_volatility: float = Field(0.25, gt=0.0)
    max_var_95_daily: float = Field(0.03, gt=0.0)
    max_beta: float = Field(1.30, gt=0.0)
    min_gross_exposure: float = Field(0.80, ge=0.0, le=1.0)
    """De-risking by moving to cash never goes below this exposure."""


class CostConfig(_Frozen):
    commission_bps: float = Field(1.0, ge=0.0)
    half_spread_bps: float = Field(2.0, ge=0.0)
    slippage_bps: float = Field(5.0, ge=0.0)


class StrategyConfig(_Frozen):
    strategy_version: str = "v0.1.0"
    composite_weights: dict[FactorFamily, float] = Field(
        default_factory=lambda: {
            "value": 0.25,
            "growth": 0.15,
            "profitability": 0.25,
            "momentum": 0.20,
            "revisions": 0.15,
        }
    )
    normalization: NormalizationConfig = NormalizationConfig()
    eligibility: EligibilityConfig = EligibilityConfig()
    signals: SignalThresholds = SignalThresholds()
    portfolio: PortfolioConfig = PortfolioConfig()
    costs: CostConfig = CostConfig()
    risk: RiskConfig = RiskConfig()

    @model_validator(mode="after")
    def _check_weights(self) -> StrategyConfig:
        if set(self.composite_weights) != set(FACTOR_FAMILIES):
            raise ValueError(f"composite_weights must define exactly {FACTOR_FAMILIES}")
        if any(w < 0 for w in self.composite_weights.values()):
            raise ValueError("composite_weights must be non-negative")
        if not math.isclose(sum(self.composite_weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError("composite_weights must sum to 1")
        return self


def load_strategy_config(path: str | Path) -> StrategyConfig:
    with Path(path).open(encoding="utf-8") as handle:
        return StrategyConfig.model_validate(yaml.safe_load(handle))


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "strategies" / "v0.1.0.yaml"


def default_strategy_config() -> StrategyConfig:
    if DEFAULT_CONFIG_PATH.exists():
        return load_strategy_config(DEFAULT_CONFIG_PATH)
    return StrategyConfig()
