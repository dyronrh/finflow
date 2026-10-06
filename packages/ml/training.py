"""Walk-forward training of cross-sectional ranking models (README §12.2, §13).

At every decision date ``t`` in the test schedule the model in use was fitted
only on rows whose label was already known before ``t`` (``label_known_at <
t``), inside a rolling window of ``train_years``. Models are refitted every
``retrain_months``. The resulting predictions are therefore out-of-sample at
every date and can be backtested with the same engine as the rule strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml.dataset import feature_columns

MODEL_TYPES = ("ridge", "gbm")


def make_model(kind: str, random_state: int = 42):
    from sklearn.ensemble import HistGradientBoostingRegressor  # optional dependency
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline

    if kind == "ridge":
        return make_pipeline(SimpleImputer(strategy="constant", fill_value=0.5), Ridge(alpha=10.0))
    if kind == "gbm":
        # Shallow, heavily regularised trees: cross-sectional return signals are
        # weak, and deep trees mostly memorise noise.
        return HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.03,
            max_leaf_nodes=15,
            min_samples_leaf=200,
            l2_regularization=1.0,
            random_state=random_state,
        )
    raise ValueError(f"unknown model type {kind!r}; choose from {MODEL_TYPES}")


def hyperparameters(model) -> dict[str, object]:
    params = model.get_params()
    return {k: v for k, v in params.items() if isinstance(v, int | float | str | bool | None)}


def daily_ic(frame: pd.DataFrame, score: str, target: str = "forward_return") -> pd.Series:
    """Spearman IC per date."""

    def ic(g: pd.DataFrame) -> float:
        ok = g[score].notna() & g[target].notna()
        if ok.sum() < 10:
            return np.nan
        return float(g.loc[ok, score].rank().corr(g.loc[ok, target].rank()))

    return frame.groupby("date")[[score, target]].apply(ic).dropna()


def ic_summary(ic: pd.Series) -> dict[str, float]:
    if ic.empty:
        return {
            "mean_ic": float("nan"),
            "t_stat": float("nan"),
            "pct_positive": float("nan"),
            "periods": 0,
        }
    return {
        "mean_ic": float(ic.mean()),
        "t_stat": float(ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic)))
        if len(ic) > 1
        else float("nan"),
        "pct_positive": float((ic > 0).mean()),
        "periods": len(ic),
    }


def train_rows(panel: pd.DataFrame, as_of: pd.Timestamp, train_years: float) -> pd.DataFrame:
    """Rows usable to fit a model for a decision at ``as_of`` (purged, rolling)."""
    start = as_of - pd.DateOffset(months=round(train_years * 12))
    usable = panel["target"].notna() & (panel["label_known_at"] < as_of) & (panel["date"] >= start)
    return panel[usable]


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    """date, security_id, ml_score (0–100), model_fit_date."""
    fits: pd.DataFrame
    """One row per refit: fit_date, train_start, train_end, n_rows."""
    model_type: str
    feature_names: list[str] = field(default_factory=feature_columns)


def walk_forward_predict(
    panel: pd.DataFrame,
    model_type: str = "gbm",
    train_years: float = 5.0,
    retrain_months: int = 3,
    min_train_rows: int = 2_000,
    random_state: int = 42,
) -> WalkForwardResult:
    features = feature_columns()
    dates = sorted(panel["date"].unique())
    preds, fits = [], []
    model, last_fit = None, None
    for date in dates:
        date = pd.Timestamp(date)
        due = last_fit is None or date >= last_fit + pd.DateOffset(months=retrain_months)
        if due:
            rows = train_rows(panel, date, train_years)
            if len(rows) >= min_train_rows:
                model = make_model(model_type, random_state)
                model.fit(rows[features], rows["target"])
                last_fit = date
                fits.append(
                    {
                        "fit_date": date,
                        "train_start": rows["date"].min(),
                        "train_end": rows["date"].max(),
                        "label_known_until": rows["label_known_at"].max(),
                        "n_rows": len(rows),
                    }
                )
        if model is None:
            continue
        today = panel[panel["date"] == date]
        raw = pd.Series(model.predict(today[features]), index=today.index)
        preds.append(
            pd.DataFrame(
                {
                    "date": date,
                    "security_id": today["security_id"],
                    "ml_score": 100.0 * (raw.rank() - 0.5) / raw.count(),
                    "model_fit_date": last_fit,
                }
            )
        )
    predictions = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    return WalkForwardResult(predictions, pd.DataFrame(fits), model_type)


def fit_final(
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    model_type: str,
    train_years: float,
    random_state: int = 42,
):
    rows = train_rows(panel, as_of, train_years)
    model = make_model(model_type, random_state)
    model.fit(rows[feature_columns()], rows["target"])
    return model, rows


def permutation_importance_ic(
    model, rows: pd.DataFrame, n_repeats: int = 3, random_state: int = 42
) -> dict[str, float]:
    """Drop in mean IC when one feature is shuffled within each date.

    Shuffling *within* dates keeps the cross-sectional nature of the problem,
    which plain row-wise permutation would break.
    """
    rng = np.random.default_rng(random_state)
    features = feature_columns()
    base_frame = rows.assign(score=model.predict(rows[features]))
    base = daily_ic(base_frame, "score").mean()
    out: dict[str, float] = {}
    for feature in features:
        drops = []
        for _ in range(n_repeats):
            shuffled = rows.copy()
            shuffled[feature] = shuffled.groupby("date")[feature].transform(
                lambda s: s.sample(frac=1.0, random_state=int(rng.integers(1e9))).to_numpy()
            )
            score = model.predict(shuffled[features])
            drops.append(base - daily_ic(shuffled.assign(score=score), "score").mean())
        out[feature] = float(np.mean(drops))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
