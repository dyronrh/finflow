"""Use model predictions inside the rule-based decision pipeline (README §13.1–13.2).

The ML score replaces (or blends with) the composite score; every hard rule
still applies afterwards: eligibility, risk flags, label thresholds, portfolio
limits, risk engine. ML never trades on its own.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from quant_core.config import StrategyConfig
from quant_core.factors.normalization import cross_sectional_percentile
from quant_core.signals.rules import label_signals

SignalsFor = Callable[[pd.Timestamp, StrategyConfig], pd.DataFrame]


def ml_signal_provider(
    base: SignalsFor, predictions: pd.DataFrame, blend: float = 1.0
) -> SignalsFor:
    """``blend`` = weight of the ML score (1.0 = pure ML ranking)."""
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be within [0, 1]")
    by_date = {
        pd.Timestamp(d): g.set_index("security_id")["ml_score"]
        for d, g in predictions.groupby("date")
    }

    def provider(as_of: pd.Timestamp, config: StrategyConfig) -> pd.DataFrame:
        signals = base(as_of, config)
        scores = by_date.get(pd.Timestamp(as_of))
        if scores is None or signals.empty:
            # No model fitted yet for this date: no ML decision (engine holds).
            return signals.iloc[0:0]
        out = signals.drop(columns=["decision"], errors="ignore").copy()
        ml = out["security_id"].map(scores)
        out["rule_composite_score"] = out["composite_score"]
        out["ml_score"] = ml
        out["composite_score"] = (blend * ml + (1.0 - blend) * out["composite_score"]).fillna(
            out["composite_score"]
        )
        out["composite_percentile"] = cross_sectional_percentile(out["composite_score"])
        out["explanation"] = [
            [*lines, f"Score ML {m:.0f}/100 (mezcla {blend:.0%})"] if pd.notna(m) else lines
            for lines, m in zip(out["explanation"], ml, strict=True)
        ]
        labeled = label_signals(out, config)
        return labeled.sort_values(
            ["composite_score", "security_id"], ascending=[False, True]
        ).reset_index(drop=True)

    return provider
