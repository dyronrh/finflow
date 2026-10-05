"""Point-in-time access helpers (README §2.1).

Every record carries ``available_at``: the moment the data became knowable.
A decision at ``as_of`` may only see rows with ``available_at <= as_of``.
"""

from __future__ import annotations

import pandas as pd


class PointInTimeViolation(AssertionError):
    """Raised when data from the future reaches a decision."""


def as_of_view(
    frame: pd.DataFrame, as_of: pd.Timestamp, available_column: str = "available_at"
) -> pd.DataFrame:
    """Rows that were already available at ``as_of``."""
    return frame[frame[available_column] <= pd.Timestamp(as_of)]


def latest_as_of(
    frame: pd.DataFrame,
    as_of: pd.Timestamp,
    key: str = "security_id",
    order_column: str = "available_at",
    available_column: str = "available_at",
) -> pd.DataFrame:
    """Most recent available row per ``key`` as of ``as_of``."""
    visible = as_of_view(frame, as_of, available_column)
    if visible.empty:
        return visible
    return (
        visible.sort_values([key, order_column])
        .groupby(key, as_index=False, sort=True)
        .tail(1)
        .reset_index(drop=True)
    )


def assert_point_in_time(
    frame: pd.DataFrame, as_of: pd.Timestamp, available_column: str = "available_at"
) -> None:
    future = frame[frame[available_column] > pd.Timestamp(as_of)]
    if not future.empty:
        raise PointInTimeViolation(
            f"{len(future)} rows have {available_column} after as_of={pd.Timestamp(as_of).date()}"
        )
