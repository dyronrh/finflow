"""Point-in-time training panel for ranking models (README §13.1).

One row per (rebalance date, eligible security):

* features: sector-relative percentiles of every raw factor input, the five
  family scores, volatility and liquidity percentiles — all computed with data
  available at the decision date (same pipeline as the rule-based strategy);
* target: cross-sectional percentile of the forward return from the open
  after this rebalance to the open after the next one (the holding period);
* ``label_known_at``: the date the target becomes observable. Training for a
  decision at ``t`` may only use rows with ``label_known_at < t`` (purging).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from data_platform.market import MarketData
from quant_core.config import FACTOR_FAMILIES, StrategyConfig
from quant_core.factors.definitions import FACTOR_DEFINITIONS
from quant_core.factors.normalization import (
    cross_sectional_percentile,
    winsorized_sector_percentile,
)

FEATURE_SCHEMA_VERSION = "fs_v1"
SignalsFor = Callable[[pd.Timestamp, StrategyConfig], pd.DataFrame]


def feature_columns() -> list[str]:
    raw = [f"pct_{f}" for features in FACTOR_DEFINITIONS.values() for f in features]
    families = [f"{f}_score" for f in FACTOR_FAMILIES]
    return [*raw, *families, "pct_volatility_63d", "pct_adv_usd"]


def features_from_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """Model inputs for one cross-section (rows aligned with ``signals``)."""
    out = pd.DataFrame(index=signals.index)
    for features in FACTOR_DEFINITIONS.values():
        for feature, higher_is_better in features.items():
            if feature in signals.columns and signals[feature].notna().sum() >= 5:
                out[f"pct_{feature}"] = winsorized_sector_percentile(
                    signals, feature, higher_is_better=higher_is_better
                )
            else:
                out[f"pct_{feature}"] = np.nan
    for family in FACTOR_FAMILIES:
        out[f"{family}_score"] = signals[f"{family}_score"] / 100.0
    out["pct_volatility_63d"] = cross_sectional_percentile(signals["volatility_63d"])
    out["pct_adv_usd"] = cross_sectional_percentile(signals["adv_usd"])
    return out[feature_columns()]


def build_panel(
    market: MarketData,
    config: StrategyConfig,
    signals_for: SignalsFor,
    schedule: list[pd.Timestamp],
) -> pd.DataFrame:
    position = {d: i for i, d in enumerate(market.dates)}
    rows = []
    for this, nxt in zip(schedule, [*schedule[1:], None], strict=True):
        signals = signals_for(this, config)
        if signals.empty or "composite_score" not in signals.columns:
            continue
        frame = features_from_signals(signals)
        frame.insert(0, "security_id", signals["security_id"].to_numpy())
        frame.insert(0, "date", this)
        frame["composite_score"] = signals["composite_score"].to_numpy()

        target = pd.Series(np.nan, index=signals.index)
        known_at = pd.NaT
        if nxt is not None and position[nxt] + 1 < len(market.dates):
            i0, i1 = position[this] + 1, position[nxt] + 1
            p0 = market.open.iloc[i0].reindex(signals["security_id"]).to_numpy()
            p1 = market.open.iloc[i1].reindex(signals["security_id"]).to_numpy()
            fwd = pd.Series(p1 / p0 - 1.0, index=signals.index)
            target = cross_sectional_percentile(fwd)
            frame["forward_return"] = fwd.to_numpy()
            known_at = market.dates[i1]
        else:
            frame["forward_return"] = np.nan
        frame["target"] = target.to_numpy()
        frame["label_known_at"] = known_at
        rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=["date", "security_id", *feature_columns(), "target"])
    return pd.concat(rows, ignore_index=True)
