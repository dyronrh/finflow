"""Read-only views that feed the web dashboard.

Everything here is derived from the same ResearchService used by the API and
pipelines, so the UI never shows numbers the strategy would not produce.
Paper-trading data is exposed read-only: approving and executing stay in the
CLI until the API has authentication and roles (README §15).
"""

from __future__ import annotations

import math
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from apps.api.services.research import ResearchService, UnknownSecurity, factor_scores
from backtesting.engine import rebalance_dates, run_backtest
from backtesting.metrics import performance_summary
from quant_core.config import FACTOR_FAMILIES
from quant_core.portfolio.construction import build_target_weights
from quant_core.signals.rules import Decision

ROOT = Path(__file__).resolve().parents[3]
PAPER_DIR = ROOT / "data" / "local_dev_only" / "paper"


def clean(value):
    """JSON-safe: NaN/inf → None, numpy scalars → Python, timestamps → ISO dates."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [clean(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _series_points(series: pd.Series, decimals: int = 4) -> list[dict]:
    s = series.dropna()
    return [
        {"time": d.strftime("%Y-%m-%d"), "value": round(float(v), decimals)} for d, v in s.items()
    ]


class DashboardService:
    def __init__(self, research: ResearchService, source: str) -> None:
        self.research = research
        self.market = research.market
        self.config = research.config
        self.source = source
        self._lock = threading.Lock()
        self._backtest = lru_cache(maxsize=4)(self._run_backtest)

    # ---------------------------------------------------------------- overview
    def overview(self) -> dict:
        as_of = self.research.resolve_as_of(None)
        signals = self.research.signals(as_of)
        counts = signals["decision"].value_counts().to_dict() if len(signals) else {}
        if len(signals):
            positive = signals["decision"].isin(["STRONG_LONG", "LONG", "WATCH"])
            sector = (
                signals.assign(positive=positive)
                .groupby("sector_id")
                .agg(count=("security_id", "size"), positive=("positive", "sum"))
                .reset_index()
            )
        else:
            sector = pd.DataFrame(columns=["sector_id", "count", "positive"])
        bench = (
            self.market.benchmark_close
            if self.market.benchmark_close is not None
            else (1 + self.market.close.pct_change(fill_method=None).mean(axis=1)).cumprod()
        )
        bench = bench.loc[:as_of].tail(260)
        top = signals.head(10)
        return clean(
            {
                "as_of_date": as_of,
                "data_version": self.market.data_version,
                "data_source": self.source,
                "strategy_version": self.config.strategy_version,
                "universe": self.market.metadata.get("universe", "synthetic universe"),
                "warnings": self.market.metadata.get("warnings", []),
                "securities_in_dataset": len(self.market.securities),
                "eligible": len(signals),
                "decision_counts": {d.value: int(counts.get(d.value, 0)) for d in Decision},
                "sectors": [
                    {"sector_id": r.sector_id, "count": int(r.count), "positive": int(r.positive)}
                    for r in sector.itertuples()
                ],
                "top": [self._rank_row(r) for _, r in top.iterrows()],
                "benchmark": {
                    "name": "SPY" if self.market.benchmark_close is not None else "Equal weight",
                    "points": _series_points(bench / bench.iloc[0] * 100 if len(bench) else bench),
                },
            }
        )

    def _rank_row(self, row: pd.Series) -> dict:
        return {
            "security_id": row["security_id"],
            "ticker": row["ticker"],
            "sector_id": row["sector_id"],
            "decision": row["decision"],
            "composite_score": row["composite_score"],
            "percentile": row["composite_percentile"],
            "factor_scores": factor_scores(row),
            "price": row.get("price"),
            "market_cap_usd": row.get("market_cap_usd"),
            "volatility_63d": row.get("volatility_63d"),
            "risk_flags": list(row.get("risk_flags", [])),
        }

    def securities(self) -> list[dict]:
        sec = self.market.securities
        last = self.market.close.ffill().iloc[-1]
        return clean(
            [
                {
                    "security_id": r.security_id,
                    "ticker": r.ticker,
                    "sector_id": r.sector_id,
                    "has_prices": bool(pd.notna(last.get(r.security_id))),
                }
                for r in sec.itertuples()
            ]
        )

    # ---------------------------------------------------------------- security
    def prices(self, security_id: str, start: str | None = None) -> dict:
        if security_id not in self.market.close.columns:
            raise UnknownSecurity(security_id)
        frame = self.market.display_ohlcv(security_id)
        if start:
            frame = frame.loc[pd.Timestamp(start) :]
        candles = [
            {
                "time": d.strftime("%Y-%m-%d"),
                "open": round(float(r.open), 4),
                "high": round(float(r.high), 4),
                "low": round(float(r.low), 4),
                "close": round(float(r.close), 4),
                "volume": float(r.volume) if pd.notna(r.volume) else 0.0,
            }
            for d, r in frame.iterrows()
        ]
        return {"security_id": security_id, "candles": candles}

    def score_history(self, security_id: str, months: int = 24) -> dict:
        if security_id not in set(self.market.securities["security_id"]):
            raise UnknownSecurity(security_id)
        as_of = self.research.resolve_as_of(None)
        dates = self.market.dates[self.market.dates <= as_of]
        schedule = list(rebalance_dates(dates, "monthly"))[-months:]
        if schedule and schedule[-1] != as_of:
            schedule.append(as_of)
        points = []
        for d in schedule:
            with self._lock:  # signal cache is shared across requests
                signals = self.research.signals(d)
            hit = signals[signals["security_id"] == security_id]
            if hit.empty:
                points.append({"date": d, "composite_score": None, "decision": None})
                continue
            r = hit.iloc[0]
            points.append(
                {
                    "date": d,
                    "composite_score": r["composite_score"],
                    "percentile": r["composite_percentile"],
                    "decision": r["decision"],
                    **{f: r[f"{f}_score"] for f in FACTOR_FAMILIES},
                }
            )
        return clean({"security_id": security_id, "points": points})

    # ---------------------------------------------------------------- portfolio
    def model_portfolio(self) -> dict:
        as_of = self.research.resolve_as_of(None)
        with self._lock:
            signals = self.research.signals(as_of)
        target = build_target_weights(signals, self.config.portfolio)
        target, actions = self.research._enforce_risk(target, as_of)
        risk = self.research.portfolio_risk(target, as_of)
        meta = signals.set_index("security_id")
        rc = risk.get("risk_contributions", {})
        rows = [
            {
                "security_id": sid,
                "ticker": meta.at[sid, "ticker"],
                "sector_id": meta.at[sid, "sector_id"],
                "decision": meta.at[sid, "decision"],
                "composite_score": meta.at[sid, "composite_score"],
                "weight": w,
                "risk_contribution": rc.get(sid),
                "volatility_63d": meta.at[sid, "volatility_63d"],
            }
            for sid, w in target.items()
        ]
        return clean(
            {
                "as_of_date": as_of,
                "strategy_version": self.config.strategy_version,
                "risk_enforced": self.config.risk.enforce,
                "risk_actions": actions,
                "holdings": rows,
                "cash_weight": 1.0 - float(target.sum()),
                "risk": risk["risk"],
                "breaches": risk.get("breaches", []),
                "sector_exposure": risk.get("sector_exposure", {}),
                "limits": self.config.risk.model_dump(),
                "max_sector_weight": self.config.portfolio.max_sector_weight,
                "alerts": self.research.alerts(target, as_of),
            }
        )

    # ---------------------------------------------------------------- backtest
    def backtest(self, years: int = 3) -> dict:
        return self._backtest(years)

    def _run_backtest(self, years: int) -> dict:
        end = self.research.resolve_as_of(None)
        start = max(end - pd.DateOffset(years=years), self.market.dates[0] + pd.DateOffset(years=1))
        with self._lock:
            result = run_backtest(
                self.market,
                self.config,
                start,
                end,
                signal_provider=lambda d, _cfg: self.research.signals(d),
            )
        equity = result.equity / result.equity.iloc[0] * 100
        bench = result.benchmark_equity / result.benchmark_equity.iloc[0] * 100
        series = {"strategy": _series_points(equity, 3), "equal_weight": _series_points(bench, 3)}
        if self.market.benchmark_close is not None:
            spy = self.market.benchmark_close.reindex(equity.index).ffill()
            series["spy"] = _series_points(spy / spy.iloc[0] * 100, 3)
        drawdown = equity / equity.cummax() - 1.0
        rb = result.rebalances
        return clean(
            {
                "start": equity.index[0],
                "end": equity.index[-1],
                "strategy_version": self.config.strategy_version,
                "series": series,
                "drawdown": _series_points(drawdown, 4),
                "benchmark_drawdown": _series_points(bench / bench.cummax() - 1.0, 4),
                "summary": result.summary,
                "benchmark_summary": performance_summary(result.benchmark_equity),
                "rebalances": [
                    {
                        "date": r["signal_date"],
                        "turnover": r.get("planned_turnover"),
                        "holdings": r.get("n_targets"),
                        "volatility": r.get("ex_ante_volatility"),
                    }
                    for r in rb.to_dict("records")
                ],
                "metadata": result.metadata,
            }
        )

    # ---------------------------------------------------------------- paper
    def paper(self) -> dict:
        from paper_trading.controls import TradingControls
        from paper_trading.ledger import Ledger

        ledgers = sorted(PAPER_DIR.glob("ledger_*.sqlite")) if PAPER_DIR.exists() else []
        preferred = [p for p in ledgers if self.source in p.name] or ledgers
        controls = TradingControls.from_env().state()
        if not preferred:
            return {"configured": False, "controls": controls}
        ledger = Ledger(preferred[-1])
        state_file = PAPER_DIR / f"sim_broker_{self.source}.json"
        positions, account = [], None
        if state_file.exists():
            import json

            state = json.loads(state_file.read_text())
            traded = (
                (
                    self.market.price_close
                    if self.market.price_close is not None
                    else self.market.close
                )
                .ffill()
                .iloc[-1]
            )
            for sid, qty in state["positions"].items():
                if abs(qty) < 1e-9:
                    continue
                price = traded.get(sid)
                positions.append(
                    {
                        "security_id": sid,
                        "qty": qty,
                        "price": price,
                        "value": qty * price if price is not None else None,
                    }
                )
            invested = sum(p["value"] or 0.0 for p in positions)
            account = {"cash": state["cash"], "equity": state["cash"] + invested}
            for p in positions:
                p["weight"] = (p["value"] or 0.0) / account["equity"] if account["equity"] else None
        proposals = ledger.proposals()
        for p in proposals:
            p["approval"] = ledger.approval(p["id"])
        return clean(
            {
                "configured": True,
                "ledger": preferred[-1].name,
                "controls": controls,
                "account": account,
                "positions": sorted(positions, key=lambda p: -(p["value"] or 0)),
                "proposals": proposals[::-1],
                "audit": ledger.audit_trail()[-50:][::-1],
                # One point per day (last snapshot of the day).
                "equity_history": [
                    {"time": d, "value": v}
                    for d, v in {t[:10]: v for t, v in ledger.equity_history()}.items()
                ],
            }
        )

    def paper_proposal(self, proposal_id: str) -> dict:
        from paper_trading.ledger import Ledger

        for path in sorted(PAPER_DIR.glob("ledger_*.sqlite")):
            ledger = Ledger(path)
            try:
                p = ledger.proposal(proposal_id)
            except KeyError:
                continue
            p["orders_sent"] = ledger.orders(proposal_id)
            p["audit"] = ledger.audit_trail(proposal_id)
            return clean(p)
        raise KeyError(proposal_id)


@lru_cache(maxsize=1)
def get_dashboard_service() -> DashboardService:
    import os

    from apps.api.services.research import get_research_service

    return DashboardService(
        get_research_service(), os.environ.get("FINFLOW_DATA_SOURCE", "synthetic")
    )
