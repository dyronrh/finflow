"""Factor and composite scoring (README §8.2–§8.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_core.config import FACTOR_FAMILIES, StrategyConfig
from quant_core.factors.definitions import FACTOR_DEFINITIONS
from quant_core.factors.normalization import (
    cross_sectional_percentile,
    winsorized_sector_percentile,
)

REQUIRED_COLUMNS = ("security_id", "sector_id")


def compute_factor_scores(features: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Score one cross-section (one ``as_of_date``).

    Returns the input columns plus, for each family, ``{family}_score`` in
    [0, 100] and ``{family}_percentile`` in (0, 1), and ``composite_score``,
    ``composite_percentile`` and ``data_coverage``.

    Scoring must run on the *eligible* universe only, so that percentiles are
    relative to investable peers.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in features.columns]
    if missing:
        raise ValueError(f"features missing required columns: {missing}")
    if features["security_id"].duplicated().any():
        raise ValueError("duplicate security_id in cross-section")

    norm = config.normalization
    out = features.copy()

    for family in FACTOR_FAMILIES:
        definitions = FACTOR_DEFINITIONS[family]
        percentiles = pd.DataFrame(
            {
                feature: (
                    winsorized_sector_percentile(
                        out,
                        feature,
                        lower_quantile=norm.lower_quantile,
                        upper_quantile=norm.upper_quantile,
                        higher_is_better=higher_is_better,
                        min_group_size=norm.min_group_size,
                    )
                    if feature in out.columns
                    else pd.Series(np.nan, index=out.index)
                )
                for feature, higher_is_better in definitions.items()
            }
        )
        coverage = percentiles.notna().mean(axis=1)
        score = 100.0 * percentiles.mean(axis=1, skipna=True)
        out[f"{family}_score"] = score.where(coverage >= norm.min_family_coverage)
        out[f"{family}_coverage"] = coverage

    weights = pd.Series(config.composite_weights, dtype=float)
    family_scores = out[[f"{f}_score" for f in FACTOR_FAMILIES]].set_axis(
        list(FACTOR_FAMILIES), axis=1
    )
    available = family_scores.notna()
    weight_coverage = available.mul(weights, axis=1).sum(axis=1)
    weighted_sum = family_scores.fillna(0.0).mul(weights, axis=1).sum(axis=1)
    composite = (weighted_sum / weight_coverage.replace(0.0, np.nan)).where(
        weight_coverage >= norm.min_composite_coverage
    )

    out["data_coverage"] = weight_coverage
    out["composite_score"] = composite.clip(0.0, 100.0)
    out["composite_percentile"] = cross_sectional_percentile(out["composite_score"])
    for family in FACTOR_FAMILIES:
        out[f"{family}_percentile"] = cross_sectional_percentile(out[f"{family}_score"])

    return out
