"""Request/response models for the public API (README §14)."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class FactorScores(BaseModel):
    value: float | None
    growth: float | None
    profitability: float | None
    momentum: float | None
    revisions: float | None


class RankingItem(BaseModel):
    security_id: str
    ticker: str
    sector_id: str
    decision: str
    composite_score: float
    percentile: float
    factor_scores: FactorScores
    risk_flags: list[str]
    explanation: list[str]


class RankingsResponse(BaseModel):
    as_of_date: date
    strategy_version: str
    data_version: str
    universe: str
    eligible_count: int
    items: list[RankingItem]


class ScoreChange(BaseModel):
    lookback: str
    as_of_date: date
    composite_score: float | None
    delta: float | None


class SecurityAnalysis(BaseModel):
    as_of_date: date
    strategy_version: str
    data_version: str
    security_id: str
    ticker: str
    sector_id: str
    industry_id: str
    eligible: bool
    ineligibility_reasons: list[str]
    price: float
    performance: dict[str, float | None]
    decision: str | None
    composite_score: float | None
    composite_percentile: float | None
    factor_scores: FactorScores | None
    sector_median_scores: FactorScores | None
    score_changes: list[ScoreChange]
    risk_flags: list[str]
    missing_data: list[str]
    explanation: list[str]


class RebalanceProposalRequest(BaseModel):
    as_of: date | None = None
    current_nav: float = Field(100_000.0, gt=0)
    holdings: dict[str, float] = Field(
        default_factory=dict,
        description="Current weights by security_id; the remainder is cash.",
    )


class ProposedOrder(BaseModel):
    security_id: str
    ticker: str
    side: str
    current_weight: float
    target_weight: float
    delta_weight: float
    final_weight: float
    estimated_notional_usd: float
    estimated_cost_usd: float
    reason: list[str]


class RebalanceProposal(BaseModel):
    rebalance_id: str
    portfolio_id: str
    status: str
    as_of_date: date
    strategy_version: str
    data_version: str
    current_nav: float
    estimated_turnover: float
    turnover_capped: bool
    estimated_cost_usd: float
    orders: list[ProposedOrder]
    sector_exposure_before: dict[str, float]
    sector_exposure_after: dict[str, float]
    note: str
