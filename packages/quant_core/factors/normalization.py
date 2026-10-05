"""Cross-sectional normalization of raw features (README §8.2)."""

from __future__ import annotations

import pandas as pd


def winsorized_sector_percentile(
    dataframe: pd.DataFrame,
    feature_column: str,
    sector_column: str = "sector_id",
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    higher_is_better: bool = True,
    min_group_size: int = 5,
) -> pd.Series:
    """Percentile of a feature within its sector, in the open interval (0, 1).

    1. Winsorize the feature at universe-wide quantiles to cap outliers.
    2. Flip the sign when lower values are better.
    3. Rank inside each sector; sectors with fewer than ``min_group_size``
       valid observations fall back to the universe-wide rank.

    Missing values stay missing so coverage rules can be applied downstream.
    """
    values = pd.to_numeric(dataframe[feature_column], errors="coerce").astype(float)
    valid = values.dropna()
    if valid.empty:
        return pd.Series(float("nan"), index=dataframe.index, name=feature_column)

    clipped = values.clip(valid.quantile(lower_quantile), valid.quantile(upper_quantile))
    if not higher_is_better:
        clipped = -clipped

    sectors = dataframe[sector_column]
    sector_rank = clipped.groupby(sectors).rank(method="average")
    sector_count = clipped.groupby(sectors).transform("count")
    sector_pct = (sector_rank - 0.5) / sector_count

    universe_pct = (clipped.rank(method="average") - 0.5) / clipped.count()

    result = sector_pct.where(sector_count >= min_group_size, universe_pct)
    return result.where(values.notna()).rename(feature_column)


def cross_sectional_percentile(values: pd.Series) -> pd.Series:
    """Universe-wide percentile in (0, 1); NaN stays NaN."""
    return (values.rank(method="average") - 0.5) / values.count()
