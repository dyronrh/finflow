import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from backtesting.engine import rebalance_dates, run_backtest
from backtesting.tuning import SnapshotCache
from ml.dataset import build_panel, feature_columns
from ml.monitoring import drift_report, psi
from ml.registry import ModelMetadata, ModelRegistry
from ml.strategy import ml_signal_provider
from ml.training import (
    fit_final,
    permutation_importance_ic,
    train_rows,
    walk_forward_predict,
)


@pytest.fixture(scope="module")
def panel_and_cache(market, config):
    cache = SnapshotCache(market)
    dates = market.dates[market.dates >= "2023-01-01"]
    schedule = list(rebalance_dates(dates, "monthly"))
    return build_panel(market, config, cache.signals, schedule), cache


def test_panel_targets_are_known_only_after_the_decision(panel_and_cache, market):
    panel, _ = panel_and_cache
    assert set(feature_columns()) <= set(panel.columns)
    labelled = panel[panel["target"].notna()]
    assert (labelled["label_known_at"] > labelled["date"]).all()
    assert labelled["target"].between(0, 1).all()
    # The last date has no next period yet: no label.
    assert panel.loc[panel["date"] == panel["date"].max(), "target"].isna().all()


def test_training_rows_are_purged(panel_and_cache):
    panel, _ = panel_and_cache
    as_of = pd.Timestamp("2024-06-28")
    rows = train_rows(panel, as_of, train_years=5)
    assert len(rows) > 0
    assert (rows["label_known_at"] < as_of).all()
    # The rebalance right before as_of has a label that resolves after as_of.
    assert not ((rows["date"] < as_of) & (rows["label_known_at"] >= as_of)).any()


def test_walk_forward_predictions_never_use_future_labels(panel_and_cache):
    panel, _ = panel_and_cache
    wf = walk_forward_predict(panel, "ridge", train_years=1, retrain_months=2, min_train_rows=200)
    assert not wf.predictions.empty
    for fit in wf.fits.itertuples():
        assert fit.label_known_until < fit.fit_date
    merged = wf.predictions.merge(wf.fits, left_on="model_fit_date", right_on="fit_date")
    assert (merged["model_fit_date"] <= merged["date"]).all()
    assert wf.predictions["ml_score"].between(0, 100).all()


def test_ml_provider_keeps_hard_rules(panel_and_cache, market, config):
    panel, cache = panel_and_cache
    wf = walk_forward_predict(panel, "ridge", train_years=1, retrain_months=2, min_train_rows=200)
    date = pd.Timestamp(wf.predictions["date"].max())
    provider = ml_signal_provider(cache.signals, wf.predictions, blend=1.0)
    rule = cache.signals(date, config)
    ml = provider(date, config).set_index("security_id")
    # Same eligible universe; risk-flagged names stay AVOID whatever the model says.
    assert set(ml.index) == set(rule["security_id"])
    flagged = rule.loc[rule["risk_flag"] == 1, "security_id"]
    assert (ml.loc[flagged, "decision"] == "AVOID").all()
    assert "ml_score" in ml.columns
    # Before the first fit there is no ML decision.
    early = provider(pd.Timestamp("2023-01-31"), config)
    assert early.empty
    result = run_backtest(market, config, "2023-06-01", "2024-12-31", signal_provider=provider)
    assert np.isfinite(result.equity).all()


def test_registry_lifecycle(tmp_path, panel_and_cache):
    panel, _ = panel_and_cache
    model, rows = fit_final(panel, pd.Timestamp("2024-09-30"), "ridge", 2)
    importance = permutation_importance_ic(model, rows.tail(400), n_repeats=1)
    assert set(importance) == set(feature_columns())
    reg = ModelRegistry(tmp_path)
    meta = ModelMetadata(
        model_name="ranker_ridge",
        model_version=reg.next_version("ranker_ridge"),
        model_type="ridge",
        training_data_version="test",
        feature_schema_version="fs_v1",
        feature_names=feature_columns(),
        train_start_date="2023-01-31",
        train_end_date="2024-08-30",
        validation_dates=[],
        test_dates=[],
        metrics={},
        feature_importance=importance,
        hyperparameters={},
        git_commit="abc",
    )
    reg.register(model, meta)
    with pytest.raises(FileExistsError):
        reg.register(model, meta)
    with pytest.raises(PermissionError):  # pending review: not usable
        reg.load("ranker_ridge", "v001")
    reg.set_status("ranker_ridge", "v001", "APPROVED", by="analyst@example")
    loaded, m = reg.load("ranker_ridge", "v001")
    assert m.approved_by == "analyst@example" and m.deployed_at
    np.testing.assert_allclose(
        loaded.predict(rows[feature_columns()].head(5)),
        model.predict(rows[feature_columns()].head(5)),
    )
    assert reg.next_version("ranker_ridge") == "v002"


def test_psi_and_drift_statuses():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 5000))
    assert psi(a, pd.Series(rng.normal(0, 1, 5000))) < 0.02
    assert psi(a, pd.Series(rng.normal(1.0, 1, 5000))) > 0.25
    train = pd.DataFrame({"x": a})
    recent = pd.DataFrame({"x": rng.normal(1.0, 1, 1000)})
    report = drift_report(train, recent, ["x"], a, a, training_ic=0.03, recent_ic=-0.01)
    assert report.statuses["feature_drift"] == "ALERT"
    assert report.statuses["performance"] == "ALERT"
    assert report.overall == "ALERT"
