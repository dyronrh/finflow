"""Portfolio risk metrics and limit checks (Sprint 6)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_core.config import RiskConfig
from quant_core.risk.model import TRADING_DAYS, shrunk_covariance

Z_95 = 1.6448536269514722
# E[Z | Z > z_95] for a standard normal: phi(z) / 0.05
ES_95 = 2.0627128075074257


@dataclass(frozen=True)
class Breach:
    code: str
    severity: str  # "WARNING" | "CRITICAL"
    value: float
    limit: float
    message: str
    security_id: str | None = None


@dataclass
class RiskReport:
    gross_exposure: float
    volatility_annual: float
    var_95_daily: float
    cvar_95_daily: float
    historical_var_95_daily: float
    historical_cvar_95_daily: float
    beta: float
    sector_exposure: dict[str, float]
    industry_exposure: dict[str, float]
    risk_contributions: dict[str, float]
    breaches: list[Breach] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        top = sorted(self.risk_contributions.items(), key=lambda kv: -kv[1])[:5]
        return {
            "gross_exposure": round(self.gross_exposure, 4),
            "volatility_annual": round(self.volatility_annual, 4),
            "var_95_daily": round(self.var_95_daily, 4),
            "cvar_95_daily": round(self.cvar_95_daily, 4),
            "historical_var_95_daily": round(self.historical_var_95_daily, 4),
            "historical_cvar_95_daily": round(self.historical_cvar_95_daily, 4),
            "beta": round(self.beta, 3),
            "top_sector": max(self.sector_exposure.items(), key=lambda kv: kv[1], default=None),
            "top_risk_contributors": [(k, round(v, 4)) for k, v in top],
            "breaches": [b.code for b in self.breaches],
        }


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Share of portfolio variance from each position (sums to 1)."""
    w = weights.reindex(cov.index).fillna(0.0).to_numpy()
    sigma_w = cov.to_numpy() @ w
    variance = float(w @ sigma_w)
    if variance <= 0:
        return pd.Series(0.0, index=cov.index)
    return pd.Series(w * sigma_w / variance, index=cov.index)


def compute_risk(
    weights: pd.Series,
    returns: pd.DataFrame,
    limits: RiskConfig,
    sectors: pd.Series | None = None,
    industries: pd.Series | None = None,
    benchmark_returns: pd.Series | None = None,
    max_sector_weight: float | None = None,
) -> RiskReport:
    """Ex-ante risk of ``weights`` (cash = 1 - sum) from trailing ``returns``."""
    w = weights[weights > 0].astype(float)
    gross = float(w.sum())
    if w.empty:
        return RiskReport(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, {}, {})

    rets = returns.reindex(columns=w.index)
    cov = shrunk_covariance(rets)
    variance = float(w.to_numpy() @ cov.loc[w.index, w.index].to_numpy() @ w.to_numpy())
    vol = float(np.sqrt(max(variance, 0.0)))
    daily = vol / np.sqrt(TRADING_DAYS)

    hist = rets.fillna(0.0) @ w
    hist_var = float(-hist.quantile(0.05)) if len(hist) > 20 else float("nan")
    tail = hist[hist <= hist.quantile(0.05)]
    hist_cvar = float(-tail.mean()) if len(tail) else float("nan")

    beta = float("nan")
    if benchmark_returns is not None:
        joined = pd.concat([hist, benchmark_returns.reindex(hist.index)], axis=1).dropna()
        if len(joined) > 20 and joined.iloc[:, 1].var() > 0:
            beta = float(joined.cov().iloc[0, 1] / joined.iloc[:, 1].var())

    def exposure(groups: pd.Series | None) -> dict[str, float]:
        if groups is None:
            return {}
        return w.groupby(groups.reindex(w.index).fillna("unknown")).sum().round(6).to_dict()

    rc = risk_contributions(w, cov.loc[w.index, w.index])
    report = RiskReport(
        gross_exposure=gross,
        volatility_annual=vol,
        var_95_daily=Z_95 * daily,
        cvar_95_daily=ES_95 * daily,
        historical_var_95_daily=hist_var,
        historical_cvar_95_daily=hist_cvar,
        beta=beta,
        sector_exposure=exposure(sectors),
        industry_exposure=exposure(industries),
        risk_contributions=rc.round(6).to_dict(),
    )
    report.breaches = check_limits(report, limits, max_sector_weight)
    return report


def check_limits(
    report: RiskReport, limits: RiskConfig, max_sector_weight: float | None = None
) -> list[Breach]:
    tol = 1e-6
    out: list[Breach] = []

    def add(code, value, limit, message, severity="WARNING", security_id=None):
        out.append(Breach(code, severity, float(value), float(limit), message, security_id))

    if report.volatility_annual > limits.max_portfolio_volatility + tol:
        add(
            "PORTFOLIO_VOLATILITY",
            report.volatility_annual,
            limits.max_portfolio_volatility,
            f"Volatilidad ex-ante {report.volatility_annual:.1%} sobre el límite "
            f"{limits.max_portfolio_volatility:.0%}",
            "CRITICAL",
        )
    if report.var_95_daily > limits.max_var_95_daily + tol:
        add(
            "VAR_95",
            report.var_95_daily,
            limits.max_var_95_daily,
            f"VaR 95% diario {report.var_95_daily:.2%} sobre el límite "
            f"{limits.max_var_95_daily:.2%}",
            "CRITICAL",
        )
    if np.isfinite(report.beta) and report.beta > limits.max_beta + tol:
        add("BETA", report.beta, limits.max_beta, f"Beta {report.beta:.2f} sobre {limits.max_beta}")
    if max_sector_weight is not None:
        for sector, value in report.sector_exposure.items():
            if value > max_sector_weight + tol:
                add(
                    "SECTOR_CONCENTRATION",
                    value,
                    max_sector_weight,
                    f"Sector {sector}: {value:.1%} > {max_sector_weight:.0%}",
                    security_id=None,
                )
    for industry, value in report.industry_exposure.items():
        if value > limits.max_industry_weight + tol:
            add(
                "INDUSTRY_CONCENTRATION",
                value,
                limits.max_industry_weight,
                f"Industria {industry}: {value:.1%} > {limits.max_industry_weight:.0%}",
            )
    for name, rc in report.risk_contributions.items():
        if rc > limits.max_single_name_risk_contribution + tol:
            add(
                "RISK_CONTRIBUTION",
                rc,
                limits.max_single_name_risk_contribution,
                f"{name} aporta {rc:.1%} del riesgo total "
                f"(límite {limits.max_single_name_risk_contribution:.0%})",
                security_id=name,
            )
    return out
