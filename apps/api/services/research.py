"""Research service: snapshots, signals and rebalance proposals.

Until a licensed point-in-time vendor is wired in, the service runs on the
deterministic synthetic market from ``data_platform.synthetic``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from backtesting.engine import benchmark_returns_as_of
from data_platform.features import build_feature_snapshot
from data_platform.synthetic import SyntheticMarket, generate_synthetic_market
from quant_core.config import FACTOR_FAMILIES, StrategyConfig, default_strategy_config
from quant_core.execution.costs import TransactionCostModel
from quant_core.portfolio.construction import build_target_weights, portfolio_sector_exposure
from quant_core.portfolio.rebalance import plan_rebalance
from quant_core.risk.alerts import Alert, risk_alerts, signal_alerts
from quant_core.risk.enforce import enforce_risk_limits
from quant_core.risk.metrics import compute_risk
from quant_core.risk.model import trailing_returns
from quant_core.signals.rules import apply_eligibility, generate_signals


class AsOfOutOfRange(LookupError):
    pass


class UnknownSecurity(LookupError):
    pass


class ResearchService:
    def __init__(self, market: SyntheticMarket, config: StrategyConfig) -> None:
        self.market = market
        self.config = config
        self._signals = lru_cache(maxsize=64)(self._compute_signals)
        self._features = lru_cache(maxsize=64)(self._compute_features)

    # --- snapshots -------------------------------------------------------
    def resolve_as_of(self, as_of: object | None) -> pd.Timestamp:
        if as_of is None:
            return self.market.dates[-1]
        resolved = self.market.trading_date_on_or_before(pd.Timestamp(as_of))
        if resolved is None:
            raise AsOfOutOfRange(f"no data on or before {as_of}")
        return resolved

    def _compute_features(self, as_of: pd.Timestamp) -> pd.DataFrame:
        return build_feature_snapshot(self.market, as_of)

    def _compute_signals(self, as_of: pd.Timestamp) -> pd.DataFrame:
        return generate_signals(self._features(as_of), self.config)

    def features(self, as_of: pd.Timestamp) -> pd.DataFrame:
        return self._features(as_of)

    def signals(self, as_of: pd.Timestamp) -> pd.DataFrame:
        return self._signals(as_of)

    # --- analysis --------------------------------------------------------
    def security_analysis(self, security_id: str, as_of: pd.Timestamp) -> dict[str, object]:
        features = apply_eligibility(self.features(as_of), self.config)
        row = features[features["security_id"] == security_id]
        if row.empty:
            raise UnknownSecurity(security_id)
        raw = row.iloc[0]
        signals = self.signals(as_of)
        scored = signals[signals["security_id"] == security_id]
        s = scored.iloc[0] if not scored.empty else None

        close = self.market.close[security_id].loc[:as_of]
        performance = {
            f"return_{label}": _ret(close, n) for label, n in (("1m", 21), ("3m", 63), ("12m", 252))
        }
        missing = [
            column
            for column in (
                "earnings_yield",
                "fcf_yield",
                "ev_ebitda",
                "revenue_growth",
                "eps_growth",
                "roic",
                "gross_margin",
                "fcf_margin",
                "net_debt_ebitda",
                "mom_12_1",
                "eps_rev_30d",
                "eps_rev_90d",
                "revisions_up_ratio",
            )
            if column not in raw.index or pd.isna(raw[column])
        ]

        sector_median = None
        changes = []
        if s is not None:
            peers = signals[signals["sector_id"] == s["sector_id"]]
            sector_median = {f: _f(peers[f"{f}_score"].median()) for f in FACTOR_FAMILIES}
            for label, lag in (("1w", 5), ("1m", 21), ("3m", 63)):
                past_date = self._shift(as_of, lag)
                if past_date is None:
                    continue
                past = self.signals(past_date)
                hit = past[past["security_id"] == security_id]
                past_score = _f(hit["composite_score"].iloc[0]) if not hit.empty else None
                changes.append(
                    {
                        "lookback": label,
                        "as_of_date": past_date.date(),
                        "composite_score": past_score,
                        "delta": None
                        if past_score is None
                        else _f(s["composite_score"] - past_score),
                    }
                )

        return {
            "as_of_date": as_of.date(),
            "strategy_version": self.config.strategy_version,
            "data_version": self.market.data_version,
            "security_id": security_id,
            "ticker": raw["ticker"],
            "sector_id": raw["sector_id"],
            "industry_id": raw["industry_id"],
            "eligible": bool(raw["eligible"]),
            "ineligibility_reasons": list(raw["ineligibility_reasons"]),
            "price": float(raw["price"]),
            "performance": performance,
            "decision": None if s is None else s["decision"],
            "composite_score": None if s is None else _f(s["composite_score"]),
            "composite_percentile": None if s is None else _f(s["composite_percentile"]),
            "factor_scores": None if s is None else factor_scores(s),
            "sector_median_scores": sector_median,
            "score_changes": changes,
            "risk_flags": [] if s is None else list(s["risk_flags"]),
            "missing_data": missing,
            "explanation": [] if s is None else list(s["explanation"]),
        }

    def _shift(self, as_of: pd.Timestamp, lag: int) -> pd.Timestamp | None:
        pos = self.market.dates.get_loc(as_of) - lag
        return None if pos < 0 else self.market.dates[pos]

    # --- rebalance -------------------------------------------------------
    def rebalance_proposal(
        self, as_of: pd.Timestamp, holdings: dict[str, float], nav: float
    ) -> dict[str, object]:
        known = set(self.market.securities["security_id"])
        unknown = sorted(set(holdings) - known)
        if unknown:
            raise UnknownSecurity(", ".join(unknown))
        current = pd.Series(holdings, dtype=float)
        if (current < 0).any() or current.sum() > 1.0 + 1e-9:
            raise ValueError("holdings must be long-only weights summing to at most 1")

        pcfg = self.config.portfolio
        signals = self.signals(as_of)
        target = build_target_weights(signals, pcfg, set(current.index))
        target, risk_actions = self._enforce_risk(target, as_of)
        plan = plan_rebalance(current, target, pcfg.trade_band, pcfg.max_turnover_per_rebalance)
        costs = TransactionCostModel.from_config(self.config.costs)

        meta = self.market.securities.set_index("security_id")
        by_id = signals.set_index("security_id")
        orders, total_cost = [], 0.0
        for order in plan.orders.to_dict("records"):
            if order["side"] == "HOLD" and order["reason"] == "":
                continue
            notional = order["delta_weight"] * nav
            cost = costs.cost(notional)
            total_cost += cost
            reasons = [order["reason"]]
            if order["security_id"] in by_id.index:
                sig = by_id.loc[order["security_id"]]
                reasons.append(f"Señal {sig['decision']} (score {sig['composite_score']:.1f})")
            else:
                reasons.append("Fuera del universo elegible")
            orders.append(
                {
                    **order,
                    "ticker": meta.loc[order["security_id"], "ticker"],
                    "estimated_notional_usd": round(notional, 2),
                    "estimated_cost_usd": round(cost, 2),
                    "reason": reasons,
                }
            )
        sectors = meta["sector_id"]
        return {
            "status": "PENDING_APPROVAL",
            "as_of_date": as_of.date(),
            "strategy_version": self.config.strategy_version,
            "data_version": self.market.data_version,
            "current_nav": nav,
            "estimated_turnover": plan.turnover,
            "turnover_capped": plan.turnover_capped,
            "estimated_cost_usd": round(total_cost, 2),
            "orders": orders,
            "sector_exposure_before": portfolio_sector_exposure(current, sectors).to_dict(),
            "sector_exposure_after": portfolio_sector_exposure(
                plan.final_weights, sectors
            ).to_dict(),
            "risk_before": self.portfolio_risk(current, as_of)["risk"],
            "risk_after": self.portfolio_risk(plan.final_weights, as_of)["risk"],
            "risk_actions": risk_actions,
            "alerts": self.alerts(current, as_of),
            "note": (
                "Propuesta únicamente: no se envían órdenes. "
                "La ejecución requiere aprobación humana."
            ),
        }

    # --- risk ----------------------------------------------------------------
    def _risk_inputs(self, names: list[str], as_of: pd.Timestamp):
        rcfg = self.config.risk
        meta = self.market.securities.set_index("security_id")
        returns = trailing_returns(self.market.close, as_of, names, rcfg.lookback_days)
        bench = benchmark_returns_as_of(self.market, as_of, rcfg.lookback_days)
        industries = meta["industry_id"] if "industry_id" in meta else meta["sector_id"]
        return returns, bench, meta["sector_id"], industries

    def _enforce_risk(self, target: pd.Series, as_of: pd.Timestamp) -> tuple[pd.Series, list[str]]:
        if target.empty or not self.config.risk.enforce:
            return target, []
        returns, bench, sectors, industries = self._risk_inputs(list(target.index), as_of)
        return enforce_risk_limits(
            target,
            returns,
            self.config.risk,
            sectors,
            industries,
            self.config.portfolio.max_position_weight,
            self.config.portfolio.max_sector_weight,
            bench,
        )

    def portfolio_risk(self, weights: pd.Series, as_of: pd.Timestamp) -> dict[str, object]:
        weights = weights[weights > 0]
        if weights.empty:
            return {"risk": {"gross_exposure": 0.0}, "breaches": []}
        returns, bench, sectors, industries = self._risk_inputs(list(weights.index), as_of)
        report = compute_risk(
            weights,
            returns,
            self.config.risk,
            sectors,
            industries,
            bench,
            self.config.portfolio.max_sector_weight,
        )
        return {
            "risk": report.summary(),
            "breaches": [b.__dict__ for b in report.breaches],
            "sector_exposure": report.sector_exposure,
            "risk_contributions": report.risk_contributions,
        }

    def alerts(self, weights: pd.Series, as_of: pd.Timestamp) -> list[dict[str, object]]:
        """Risk breaches of the current book plus signal changes over the last month."""
        held = set(weights[weights > 0].index)
        out: list[Alert] = []
        if held:
            returns, bench, sectors, industries = self._risk_inputs(sorted(held), as_of)
            report = compute_risk(
                weights[weights > 0],
                returns,
                self.config.risk,
                sectors,
                industries,
                bench,
                self.config.portfolio.max_sector_weight,
            )
            out.extend(risk_alerts(report))
        previous = self._shift(as_of, 21)
        if previous is not None:
            out.extend(signal_alerts(self.signals(previous), self.signals(as_of), held))
        return [a.to_dict() for a in out]


def factor_scores(row: pd.Series) -> dict[str, float | None]:
    return {f: _f(row[f"{f}_score"]) for f in FACTOR_FAMILIES}


def _f(value: object) -> float | None:
    if value is None:
        return None
    value = float(value)  # type: ignore[arg-type]
    return None if np.isnan(value) else round(value, 4)


def _ret(close: pd.Series, n: int) -> float | None:
    return None if len(close) <= n else _f(close.iloc[-1] / close.iloc[-1 - n] - 1.0)


@lru_cache(maxsize=1)
def get_research_service() -> ResearchService:
    return ResearchService(generate_synthetic_market(), default_strategy_config())
