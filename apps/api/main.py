"""Quant Portfolio Intelligence API (README §14)."""

from __future__ import annotations

import uuid
from datetime import date
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query

from apps.api.schemas.models import (
    PortfolioRiskRequest,
    PortfolioRiskResponse,
    RankingItem,
    RankingsResponse,
    RebalanceProposal,
    RebalanceProposalRequest,
    SecurityAnalysis,
)
from apps.api.services.research import (
    AsOfOutOfRange,
    ResearchService,
    UnknownSecurity,
    factor_scores,
    get_research_service,
)
from quant_core.signals.rules import Decision

app = FastAPI(
    title="Quant Portfolio Intelligence",
    version="0.1.0",
    description=(
        "Rankings multifactoriales, análisis de activos y propuestas de rebalanceo. "
        "Herramienta de investigación; no constituye asesoría financiera."
    ),
)

Service = Annotated[ResearchService, Depends(get_research_service)]


def _as_of(service: ResearchService, as_of: date | None):
    try:
        return service.resolve_as_of(as_of)
    except AsOfOutOfRange as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/rankings", response_model=RankingsResponse)
def rankings(
    service: Service,
    as_of: date | None = None,
    universe: str = "us_equities",
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    decision: Decision | None = None,
) -> RankingsResponse:
    if universe != "us_equities":
        raise HTTPException(status_code=404, detail=f"unknown universe: {universe}")
    resolved = _as_of(service, as_of)
    signals = service.signals(resolved)
    selected = signals if decision is None else signals[signals["decision"] == decision.value]
    items = [
        RankingItem(
            security_id=row["security_id"],
            ticker=row["ticker"],
            sector_id=row["sector_id"],
            decision=row["decision"],
            composite_score=round(float(row["composite_score"]), 2),
            percentile=round(float(row["composite_percentile"]), 4),
            factor_scores=factor_scores(row),
            risk_flags=list(row["risk_flags"]),
            explanation=list(row["explanation"]),
        )
        for _, row in selected.head(limit).iterrows()
    ]
    return RankingsResponse(
        as_of_date=resolved.date(),
        strategy_version=service.config.strategy_version,
        data_version=service.market.data_version,
        universe=universe,
        eligible_count=len(signals),
        items=items,
    )


@app.get("/v1/securities/{security_id}/analysis", response_model=SecurityAnalysis)
def security_analysis(
    security_id: str, service: Service, as_of: date | None = None
) -> SecurityAnalysis:
    resolved = _as_of(service, as_of)
    try:
        return SecurityAnalysis(**service.security_analysis(security_id, resolved))
    except UnknownSecurity as exc:
        raise HTTPException(status_code=404, detail=f"unknown security: {exc}") from exc


@app.post("/v1/portfolios/{portfolio_id}/rebalance/proposal", response_model=RebalanceProposal)
def rebalance_proposal(
    portfolio_id: str, request: RebalanceProposalRequest, service: Service
) -> RebalanceProposal:
    resolved = _as_of(service, request.as_of)
    try:
        proposal = service.rebalance_proposal(resolved, request.holdings, request.current_nav)
    except UnknownSecurity as exc:
        raise HTTPException(status_code=404, detail=f"unknown security: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    rebalance_id = f"rb_{resolved:%Y_%m_%d}_{uuid.uuid4().hex[:8]}"
    return RebalanceProposal(rebalance_id=rebalance_id, portfolio_id=portfolio_id, **proposal)


@app.post("/v1/portfolios/{portfolio_id}/risk", response_model=PortfolioRiskResponse)
def portfolio_risk(
    portfolio_id: str, request: PortfolioRiskRequest, service: Service
) -> PortfolioRiskResponse:
    resolved = _as_of(service, request.as_of)
    known = set(service.market.securities["security_id"])
    unknown = sorted(set(request.holdings) - known)
    if unknown:
        raise HTTPException(status_code=404, detail=f"unknown security: {', '.join(unknown)}")
    weights = pd.Series(request.holdings, dtype=float)
    risk = service.portfolio_risk(weights, resolved)
    return PortfolioRiskResponse(
        as_of_date=resolved.date(),
        strategy_version=service.config.strategy_version,
        alerts=service.alerts(weights, resolved),
        **risk,
    )
