"""Data-contract checks (README §7.4, §17.2)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ContractResult:
    dataset: str
    violations: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.violations


def validate_factor_scores(scores: pd.DataFrame, as_of: pd.Timestamp) -> ContractResult:
    """Contract for ``factor_scores_daily``."""
    violations: list[str] = []
    required = [
        "as_of_date",
        "security_id",
        "sector_id",
        "composite_score",
        "value_score",
        "growth_score",
        "profitability_score",
        "momentum_score",
        "revisions_score",
        "data_version",
    ]
    missing = [c for c in required if c not in scores.columns]
    if missing:
        return ContractResult("factor_scores_daily", (f"missing columns: {missing}",))

    if scores.duplicated(["as_of_date", "security_id"]).any():
        violations.append("duplicate primary keys (as_of_date, security_id)")
    if scores["sector_id"].isna().any():
        violations.append("sector_id is null")
    for column in [c for c in required if c.endswith("_score")]:
        values = scores[column].dropna()
        if ((values < 0) | (values > 100)).any():
            violations.append(f"{column} outside [0, 100]")
    for column in ("fundamentals_available_at", "estimates_available_at"):
        if column in scores.columns and (scores[column].dropna() > pd.Timestamp(as_of)).any():
            violations.append(f"{column} > as_of_date (look-ahead)")
    if (pd.to_datetime(scores["as_of_date"]) > pd.Timestamp(as_of)).any():
        violations.append("as_of_date in the future")
    return ContractResult("factor_scores_daily", tuple(violations))
