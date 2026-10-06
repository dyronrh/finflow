"""Alerts on risk, signals and drawdowns (README §3.1, §11.1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from quant_core.risk.metrics import RiskReport
from quant_core.signals.rules import EXIT_DECISIONS, Decision


@dataclass(frozen=True)
class Alert:
    severity: str  # INFO | WARNING | CRITICAL
    code: str
    message: str
    security_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def risk_alerts(report: RiskReport) -> list[Alert]:
    return [Alert(b.severity, b.code, b.message, b.security_id) for b in report.breaches]


def signal_alerts(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    holdings: set[str] | frozenset[str],
    score_drop_points: float = 15.0,
) -> list[Alert]:
    """Changes between two signal snapshots that matter for the portfolio."""
    alerts: list[Alert] = []
    prev = previous.set_index("security_id") if len(previous) else pd.DataFrame()
    cur = current.set_index("security_id") if len(current) else pd.DataFrame()
    exits = {d.value for d in EXIT_DECISIONS}

    for sid in sorted(holdings):
        if sid not in cur.index:
            alerts.append(
                Alert(
                    "WARNING",
                    "HOLDING_LEFT_UNIVERSE",
                    f"{sid} ya no es elegible (datos, liquidez o salida del índice)",
                    sid,
                )
            )
            continue
        decision = cur.at[sid, "decision"]
        if decision in exits and (sid not in prev.index or prev.at[sid, "decision"] not in exits):
            alerts.append(Alert("WARNING", "HOLDING_DOWNGRADED", f"{sid} pasó a {decision}", sid))
        if sid in prev.index:
            drop = prev.at[sid, "composite_score"] - cur.at[sid, "composite_score"]
            if drop >= score_drop_points:
                alerts.append(
                    Alert(
                        "INFO",
                        "SCORE_DROP",
                        f"{sid}: score bajó {drop:.1f} puntos",
                        sid,
                    )
                )
    if len(cur):
        new_strong = cur.index[cur["decision"] == Decision.STRONG_LONG.value]
        for sid in new_strong:
            if sid not in holdings and (
                sid not in prev.index or prev.at[sid, "decision"] != Decision.STRONG_LONG.value
            ):
                alerts.append(Alert("INFO", "NEW_STRONG_LONG", f"{sid} entró en STRONG_LONG", sid))
    return alerts


def drawdown_alert(equity: pd.Series, threshold: float = 0.10) -> Alert | None:
    if len(equity) < 2:
        return None
    dd = float(equity.iloc[-1] / equity.cummax().iloc[-1] - 1.0)
    if dd <= -threshold:
        return Alert(
            "CRITICAL" if dd <= -2 * threshold else "WARNING",
            "DRAWDOWN",
            f"Drawdown actual {dd:.1%} (umbral {threshold:.0%})",
        )
    return None
