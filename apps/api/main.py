"""Quant Portfolio Intelligence API (README §14)."""

from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from apps.api.schemas.models import (
    PortfolioRiskRequest,
    PortfolioRiskResponse,
    RankingItem,
    RankingsResponse,
    RebalanceProposal,
    RebalanceProposalRequest,
    SecurityAnalysis,
)
from apps.api.services.dashboard import DashboardService, get_dashboard_service
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
Dashboard = Annotated[DashboardService, Depends(get_dashboard_service)]


def _finite(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


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
            price=_finite(row.get("price")),
            market_cap_usd=_finite(row.get("market_cap_usd")),
            volatility_63d=_finite(row.get("volatility_63d")),
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


# ----------------------------------------------------------------------------- dashboard
@app.get("/v1/overview")
def overview(dashboard: Dashboard) -> dict:
    return dashboard.overview()


@app.get("/v1/securities")
def securities(dashboard: Dashboard) -> list[dict]:
    return dashboard.securities()


@app.get("/v1/securities/{security_id}/prices")
def prices(security_id: str, dashboard: Dashboard, start: date | None = None) -> dict:
    try:
        return dashboard.prices(security_id, str(start) if start else None)
    except UnknownSecurity as exc:
        raise HTTPException(status_code=404, detail=f"unknown security: {exc}") from exc


@app.get("/v1/securities/{security_id}/score-history")
def score_history(
    security_id: str, dashboard: Dashboard, months: Annotated[int, Query(ge=1, le=60)] = 24
) -> dict:
    try:
        return dashboard.score_history(security_id, months)
    except UnknownSecurity as exc:
        raise HTTPException(status_code=404, detail=f"unknown security: {exc}") from exc


@app.get("/v1/portfolio/model")
def model_portfolio(dashboard: Dashboard) -> dict:
    return dashboard.model_portfolio()


@app.get("/v1/backtest")
def backtest(dashboard: Dashboard, years: Annotated[int, Query(ge=1, le=15)] = 3) -> dict:
    return dashboard.backtest(years)


@app.get("/v1/paper")
def paper(dashboard: Dashboard) -> dict:
    """Read-only: approvals and execution stay in the CLI until the API has auth."""
    return dashboard.paper()


@app.get("/v1/paper/proposals/{proposal_id}")
def paper_proposal(proposal_id: str, dashboard: Dashboard) -> dict:
    try:
        return dashboard.paper_proposal(proposal_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown proposal {proposal_id}") from exc


# Serve the built web app (apps/web/dist) at / when present: one process for UI + API.
_WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"
if _WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=_WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        candidate = _WEB_DIST / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_WEB_DIST / "index.html")
