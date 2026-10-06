"""Turn risk limits into active constraints on target weights (Sprint 6).

Order of operations:
1. Concentration: single-name, sector and industry caps (redistribution).
2. Risk budget: positions contributing more than their share of portfolio
   variance are trimmed and the excess redistributed.
3. Exposure: if volatility, VaR or beta still exceed their limits the whole
   book is scaled towards cash, never below ``min_gross_exposure``.

Every adjustment is recorded so the proposal can explain what risk changed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_core.config import RiskConfig
from quant_core.portfolio.construction import apply_weight_caps
from quant_core.risk.metrics import compute_risk, risk_contributions
from quant_core.risk.model import shrunk_covariance


def enforce_risk_limits(
    weights: pd.Series,
    returns: pd.DataFrame,
    limits: RiskConfig,
    sectors: pd.Series,
    industries: pd.Series,
    max_position_weight: float,
    max_sector_weight: float,
    benchmark_returns: pd.Series | None = None,
) -> tuple[pd.Series, list[str]]:
    actions: list[str] = []
    w = weights[weights > 0].astype(float)
    if w.empty:
        return w, actions
    budget = float(w.sum())
    sectors = sectors.reindex(w.index).fillna("unknown")
    industries = industries.reindex(w.index).fillna("unknown")

    # 1. Concentration (alternate both group caps until both hold).
    before = w.copy()
    for _ in range(10):
        w = apply_weight_caps(w, sectors, max_position_weight, max_sector_weight) * budget
        w = (
            apply_weight_caps(w, industries, max_position_weight, limits.max_industry_weight)
            * budget
        )
        if (
            w.groupby(sectors).sum().max() <= max_sector_weight * budget + 1e-9
            and w.groupby(industries).sum().max() <= limits.max_industry_weight * budget + 1e-9
        ):
            break
    if (w - before).abs().sum() > 1e-6:
        actions.append("INDUSTRY_SECTOR_CAPS_APPLIED")

    # 2. Risk-contribution budget.
    cov = shrunk_covariance(returns.reindex(columns=w.index))
    cap = limits.max_single_name_risk_contribution
    trimmed = False
    for _ in range(50):
        rc = risk_contributions(w, cov.loc[w.index, w.index])
        over = rc[rc > cap + 1e-6]
        if over.empty or len(w) * cap < 1.0:  # infeasible budget: stop
            break
        trimmed = True
        w[over.index] *= np.sqrt(cap / over)
        w = apply_weight_caps(w, sectors, max_position_weight, max_sector_weight) * budget
        w = (
            apply_weight_caps(w, industries, max_position_weight, limits.max_industry_weight)
            * budget
        )
    if trimmed:
        actions.append("RISK_CONTRIBUTION_TRIMMED")

    # 3. Exposure scaling for volatility / VaR / beta.
    report = compute_risk(w, returns, limits, sectors, industries, benchmark_returns)
    scale = 1.0
    if report.volatility_annual > limits.max_portfolio_volatility:
        scale = min(scale, limits.max_portfolio_volatility / report.volatility_annual)
    if report.var_95_daily > limits.max_var_95_daily:
        scale = min(scale, limits.max_var_95_daily / report.var_95_daily)
    if np.isfinite(report.beta) and report.beta > limits.max_beta:
        scale = min(scale, limits.max_beta / report.beta)
    floor = limits.min_gross_exposure / max(budget, 1e-12)
    scale = max(scale, min(floor, 1.0))
    if scale < 1.0 - 1e-9:
        w = w * scale
        actions.append(f"EXPOSURE_SCALED_TO_{w.sum():.0%}")
    return w[w > 1e-12], actions
