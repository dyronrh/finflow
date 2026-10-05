import pandas as pd
import pytest

from data_platform.features import build_feature_snapshot
from data_platform.point_in_time import PointInTimeViolation, assert_point_in_time, latest_as_of


def test_latest_as_of_ignores_future_rows():
    df = pd.DataFrame(
        {
            "security_id": ["A", "A", "A"],
            "available_at": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "v": [1, 2, 3],
        }
    )
    assert latest_as_of(df, pd.Timestamp("2024-02-15"))["v"].tolist() == [2]
    with pytest.raises(PointInTimeViolation):
        assert_point_in_time(df, pd.Timestamp("2024-02-15"))


def test_filings_respect_publication_lag(market, as_of):
    snap = build_feature_snapshot(market, as_of)
    assert (snap["fundamentals_available_at"].dropna() <= as_of).all()
    assert (snap["estimates_available_at"].dropna() <= as_of).all()


def test_features_do_not_change_when_future_data_changes(market):
    """Mutating everything after as_of must not move the snapshot (no leakage)."""
    as_of = market.dates[400]
    before = build_feature_snapshot(market, as_of)

    tampered = type(market)(
        securities=market.securities,
        open=market.open.copy(),
        close=market.close.copy(),
        volume=market.volume.copy(),
        fundamentals=market.fundamentals.copy(),
        estimates=market.estimates.copy(),
        seed=market.seed,
        data_version=market.data_version,
    )
    future = tampered.close.index > as_of
    tampered.close.loc[future] *= 10
    tampered.volume.loc[future] *= 10
    tampered.fundamentals.loc[tampered.fundamentals["available_at"] > as_of, "revenue"] *= 10
    tampered.estimates.loc[tampered.estimates["available_at"] > as_of, "fy1_eps_estimate"] *= 10

    after = build_feature_snapshot(tampered, as_of)
    pd.testing.assert_frame_equal(before, after)
