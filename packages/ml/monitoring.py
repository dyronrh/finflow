"""Model monitoring: data drift, score drift, coverage and IC decay (README §13.4)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PSI_WARN, PSI_ALERT = 0.10, 0.25


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population stability index between two samples (quantile bins of ``expected``)."""
    e, a = expected.dropna(), actual.dropna()
    if len(e) < bins or len(a) < bins:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-6, None), np.clip(pa, 1e-6, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def _status(value: float, warn: float, alert: float, higher_is_worse: bool = True) -> str:
    if not np.isfinite(value):
        return "UNKNOWN"
    v = value if higher_is_worse else -value
    return "ALERT" if v >= alert else "WARN" if v >= warn else "OK"


@dataclass
class DriftReport:
    feature_psi: dict[str, float]
    coverage_change: dict[str, float]
    score_psi: float
    training_ic: float
    recent_ic: float
    statuses: dict[str, str] = field(default_factory=dict)

    @property
    def overall(self) -> str:
        values = set(self.statuses.values())
        return "ALERT" if "ALERT" in values else "WARN" if "WARN" in values else "OK"


def drift_report(
    train: pd.DataFrame,
    recent: pd.DataFrame,
    features: list[str],
    train_scores: pd.Series,
    recent_scores: pd.Series,
    training_ic: float,
    recent_ic: float,
) -> DriftReport:
    feature_psi = {f: psi(train[f], recent[f]) for f in features}
    coverage = {f: float(recent[f].notna().mean() - train[f].notna().mean()) for f in features}
    report = DriftReport(
        feature_psi=feature_psi,
        coverage_change=coverage,
        score_psi=psi(train_scores, recent_scores),
        training_ic=training_ic,
        recent_ic=recent_ic,
    )
    worst = max((v for v in feature_psi.values() if np.isfinite(v)), default=float("nan"))
    report.statuses = {
        "feature_drift": _status(worst, PSI_WARN, PSI_ALERT),
        "score_drift": _status(report.score_psi, PSI_WARN, PSI_ALERT),
        "coverage": _status(max((-v for v in coverage.values()), default=0.0), 0.05, 0.20),
        # IC decay: recent IC below zero is an alert, below half of training a warning.
        "performance": (
            "UNKNOWN"
            if not (np.isfinite(recent_ic) and np.isfinite(training_ic))
            else "ALERT"
            if recent_ic < 0
            else "WARN"
            if recent_ic < 0.5 * training_ic
            else "OK"
        ),
    }
    return report
