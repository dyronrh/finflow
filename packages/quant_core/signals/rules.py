"""Eligibility filters, decision labels and explanations (README §3.1, §9)."""

from __future__ import annotations

import json
from enum import StrEnum

import numpy as np
import pandas as pd

from quant_core.config import FACTOR_FAMILIES, StrategyConfig
from quant_core.factors.definitions import FACTOR_LABELS_ES
from quant_core.factors.normalization import cross_sectional_percentile
from quant_core.factors.scoring import compute_factor_scores


class Decision(StrEnum):
    STRONG_LONG = "STRONG_LONG"
    LONG = "LONG"
    WATCH = "WATCH"
    NEUTRAL = "NEUTRAL"
    REDUCE = "REDUCE"
    AVOID = "AVOID"


BUY_DECISIONS = frozenset({Decision.STRONG_LONG, Decision.LONG})
EXIT_DECISIONS = frozenset({Decision.REDUCE, Decision.AVOID})


def apply_eligibility(features: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Flag the investable universe and attach the reasons for exclusion."""
    rules = config.eligibility
    out = features.copy()
    # Written as "not (value >= minimum)" so that missing data fails the check.
    checks = {
        "PRICE_BELOW_MIN": ~(out["price"] >= rules.minimum_stock_price_usd),
        "MARKET_CAP_BELOW_MIN": ~(out["market_cap_usd"] >= rules.min_market_cap_usd),
        "ADV_BELOW_MIN": ~(out["adv_usd"] >= rules.min_adv_usd),
        "MISSING_SECTOR": out["sector_id"].isna(),
    }
    if "in_universe" in out.columns:
        checks["NOT_IN_UNIVERSE"] = ~out["in_universe"].astype(bool)
    reasons = pd.Series([[] for _ in range(len(out))], index=out.index, dtype=object)
    for reason, mask in checks.items():
        for idx in out.index[mask]:
            reasons.at[idx] = [*reasons.at[idx], reason]
    out["eligible"] = reasons.map(len) == 0
    out["ineligibility_reasons"] = reasons
    return out


def generate_signals(features: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Full cross-sectional pipeline: eligibility → scoring → labels.

    Returns one row per *eligible* security, sorted by composite score.
    """
    return label_signals(score_universe(features, config), config)


def scoring_key(config: StrategyConfig) -> str:
    """Identifies the config parts that change scores (not labels or sizing)."""
    parts = {
        "eligibility": config.eligibility.model_dump(),
        "normalization": config.normalization.model_dump(),
        "weights": config.composite_weights,
    }
    return json.dumps(parts, sort_keys=True)


def score_universe(features: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Eligibility + factor scores + percentiles (the expensive, cacheable part)."""
    screened = apply_eligibility(features, config)
    eligible = screened[screened["eligible"]].reset_index(drop=True)
    if eligible.empty:
        return eligible

    scored = compute_factor_scores(eligible, config)
    scored = scored[scored["composite_score"].notna()].reset_index(drop=True)
    if scored.empty:
        return scored
    # Percentiles are re-ranked on the final scored universe.
    scored["composite_percentile"] = cross_sectional_percentile(scored["composite_score"])
    for family in FACTOR_FAMILIES:
        scored[f"{family}_percentile"] = cross_sectional_percentile(scored[f"{family}_score"])
    scored["liquidity_percentile"] = cross_sectional_percentile(scored["adv_usd"])

    vol_pct = cross_sectional_percentile(scored["volatility_63d"])
    high_vol = vol_pct > config.eligibility.max_volatility_percentile
    scored["risk_flags"] = [["HIGH_VOLATILITY"] if flag else [] for flag in high_vol]
    scored["risk_flag"] = high_vol.astype(int)
    scored["explanation"] = [_explain(row) for row in scored.to_dict("records")]
    return scored.sort_values(
        ["composite_score", "security_id"], ascending=[False, True]
    ).reset_index(drop=True)


def label_signals(scored: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Apply decision thresholds to an already-scored universe (cheap)."""
    out = scored.copy()
    if out.empty:
        out["decision"] = pd.Series(dtype=str)
        return out
    out["decision"] = _decide(out, config)
    return out


def _pct(frame: pd.DataFrame, column: str) -> pd.Series:
    # Missing family scores count as neutral-low so they never pass a "≥" gate.
    return frame[column].fillna(0.0)


def _gate(frame: pd.DataFrame, column: str, minimum: float) -> pd.Series:
    """``column >= minimum``; skipped when the whole universe lacks the factor
    (e.g. no point-in-time analyst estimates), so it cannot block every name."""
    if frame[column].isna().all():
        return pd.Series(True, index=frame.index)
    return _pct(frame, column) >= minimum


def _decide(scored: pd.DataFrame, config: StrategyConfig) -> pd.Series:
    t = config.signals
    composite = _pct(scored, "composite_percentile")
    revisions_raw = scored["revisions_percentile"]

    long_eligible = (
        (composite >= t.long_composite_percentile)
        & _gate(scored, "profitability_percentile", t.long_min_profitability_percentile)
        & _gate(scored, "momentum_percentile", t.long_min_momentum_percentile)
        & _gate(scored, "revisions_percentile", t.long_min_revisions_percentile)
        & (_pct(scored, "liquidity_percentile") >= t.long_min_liquidity_percentile)
        & (scored["risk_flag"] == 0)
    )
    strong_long = long_eligible & (composite >= t.strong_long_composite_percentile)
    avoid = (composite <= t.avoid_composite_percentile) | (scored["risk_flag"] == 1)
    reduce = (composite <= t.reduce_composite_percentile) | (
        revisions_raw.notna() & (revisions_raw <= t.reduce_revisions_percentile)
    )
    watch = composite >= t.watch_composite_percentile

    labels = np.select(
        [strong_long, long_eligible, avoid, reduce, watch],
        [
            Decision.STRONG_LONG.value,
            Decision.LONG.value,
            Decision.AVOID.value,
            Decision.REDUCE.value,
            Decision.WATCH.value,
        ],
        default=Decision.NEUTRAL.value,
    )
    return pd.Series(labels, index=scored.index)


def _explain(row: dict) -> list[str]:
    lines: list[str] = []
    composite_pct = row.get("composite_percentile")
    if composite_pct is not None and not np.isnan(composite_pct):
        top = max(1, round((1.0 - composite_pct) * 100))
        if composite_pct >= 0.5:
            lines.append(f"Top {top}% del universo elegible")
        else:
            lines.append(f"Bottom {max(1, round(composite_pct * 100))}% del universo elegible")
    for family in FACTOR_FAMILIES:
        pct = row.get(f"{family}_percentile")
        if pct is None or np.isnan(pct):
            lines.append(f"Sin datos suficientes de {FACTOR_LABELS_ES[family]}")
        elif pct >= 0.80:
            lines.append(f"Fuerte en {FACTOR_LABELS_ES[family]} (percentil {pct:.0%})")
        elif pct <= 0.20:
            lines.append(f"Débil en {FACTOR_LABELS_ES[family]} (percentil {pct:.0%})")
    if "HIGH_VOLATILITY" in row.get("risk_flags", []):
        lines.append("Volatilidad realizada en el 5% más alto del universo")
    return lines
