"""Candidate selection, weighting and concentration limits (README §10)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_core.config import PortfolioConfig
from quant_core.signals.rules import BUY_DECISIONS, EXIT_DECISIONS, Decision


def select_holdings(
    signals: pd.DataFrame,
    config: PortfolioConfig,
    current_holdings: frozenset[str] | set[str] = frozenset(),
) -> pd.DataFrame:
    """Pick the names for the model portfolio.

    Order of preference (with hysteresis to limit turnover):
    1. Current holdings that are not REDUCE/AVOID.
    2. New STRONG_LONG / LONG names by composite score.
    3. WATCH, then NEUTRAL names, only to reach ``min_holdings``.

    REDUCE/AVOID names are never bought. If the universe cannot supply
    ``min_holdings`` names the portfolio is smaller and caps leave cash.
    """
    ranked = signals.sort_values(["composite_score", "security_id"], ascending=[False, True])
    decisions = ranked["decision"]
    keep = ranked["security_id"].isin(current_holdings) & ~decisions.isin(
        [d.value for d in EXIT_DECISIONS]
    )
    buys = ~keep & decisions.isin([d.value for d in BUY_DECISIONS])
    chosen = pd.concat([ranked[keep], ranked[buys]]).head(config.max_holdings)
    for filler in (Decision.WATCH, Decision.NEUTRAL):
        if len(chosen) >= config.min_holdings:
            break
        pool = ranked[~keep & (decisions == filler.value)]
        chosen = pd.concat([chosen, pool.head(config.min_holdings - len(chosen))])
    return chosen.reset_index(drop=True)


def equal_weight(holdings: pd.DataFrame) -> pd.Series:
    n = len(holdings)
    return pd.Series(1.0 / n if n else [], index=holdings["security_id"], dtype=float)


def inverse_volatility(holdings: pd.DataFrame, floor: float = 0.05) -> pd.Series:
    vol = holdings["volatility_63d"].astype(float)
    vol = vol.fillna(vol.median() if vol.notna().any() else 1.0).clip(lower=floor)
    raw = 1.0 / vol.to_numpy()
    return pd.Series(raw / raw.sum(), index=holdings["security_id"], dtype=float)


def apply_weight_caps(
    weights: pd.Series,
    sectors: pd.Series,
    max_position: float,
    max_sector: float,
    tol: float = 1e-10,
    max_iter: int = 1_000,
) -> pd.Series:
    """Enforce single-name and sector caps, redistributing excess pro rata.

    ``weights`` and ``sectors`` share the same index. If the caps make full
    investment infeasible, the remainder is left as cash (sum < 1).
    """
    if weights.empty:
        return weights
    base = weights / weights.sum()
    w = base.copy()
    for _ in range(max_iter):
        w = w.clip(upper=max_position)
        sector_total = w.groupby(sectors).sum()
        scale = (max_sector / sector_total).clip(upper=1.0)
        w = w * sectors.map(scale)

        deficit = 1.0 - w.sum()
        if deficit <= tol:
            break
        sector_room = (max_sector - w.groupby(sectors).sum()).clip(lower=0.0)
        has_room = ((max_position - w) > tol) & (sectors.map(sector_room) > tol)
        if not has_room.any():
            break
        receivers = base[has_room]
        w = w.add(deficit * receivers / receivers.sum(), fill_value=0.0)
    return w.clip(lower=0.0)


def build_target_weights(
    signals: pd.DataFrame,
    config: PortfolioConfig,
    current_holdings: frozenset[str] | set[str] = frozenset(),
) -> pd.Series:
    """Target weights (index = security_id) for the model portfolio."""
    holdings = select_holdings(signals, config, current_holdings)
    if holdings.empty:
        return pd.Series(dtype=float)
    raw = (
        inverse_volatility(holdings)
        if config.weighting == "inverse_volatility"
        else equal_weight(holdings)
    )
    sectors = pd.Series(holdings["sector_id"].to_numpy(), index=holdings["security_id"])
    capped = apply_weight_caps(raw, sectors, config.max_position_weight, config.max_sector_weight)
    return capped[capped > 0].sort_values(ascending=False)


def portfolio_sector_exposure(weights: pd.Series, sectors: pd.Series) -> pd.Series:
    aligned = sectors.reindex(weights.index)
    return weights.groupby(aligned).sum().sort_values(ascending=False)


def herfindahl(weights: pd.Series) -> float:
    return float(np.square(weights.to_numpy()).sum())
