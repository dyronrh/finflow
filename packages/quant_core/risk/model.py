"""Risk model: trailing returns and a shrunk covariance matrix (README §10.2, §12.3).

All inputs are sliced up to the decision date, so risk estimates are as
point-in-time as the prices they come from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def trailing_returns(
    close: pd.DataFrame,
    as_of: pd.Timestamp,
    names: list[str] | pd.Index,
    lookback_days: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Daily simple returns for ``names`` over the ``lookback_days`` up to ``as_of``."""
    names = [n for n in names if n in close.columns]
    window = close.loc[: pd.Timestamp(as_of), names].tail(lookback_days + 1)
    return window.pct_change(fill_method=None).iloc[1:]


def shrunk_covariance(returns: pd.DataFrame, min_coverage: float = 0.6) -> pd.DataFrame:
    """Annualised Ledoit-Wolf covariance (shrinkage towards a scaled identity).

    Names with too little history (recent listings) get the cross-sectional
    median variance and zero correlation instead of a noisy estimate.
    """
    if returns.empty:
        return pd.DataFrame()
    coverage = returns.notna().mean()
    good = coverage.index[coverage >= min_coverage]
    x = returns[good].fillna(0.0).to_numpy()
    n, p = x.shape
    cov = pd.DataFrame(np.nan, index=returns.columns, columns=returns.columns)
    if p and n > 1:
        xc = x - x.mean(axis=0)
        sample = xc.T @ xc / n
        mu = np.trace(sample) / p
        target = mu * np.eye(p)
        d2 = np.sum((sample - target) ** 2) / p
        b2_bar = sum(np.sum((np.outer(row, row) - sample) ** 2) for row in xc) / (n**2 * p)
        shrink = 0.0 if d2 <= 0 else min(b2_bar, d2) / d2
        sigma = shrink * target + (1.0 - shrink) * sample
        cov.loc[good, good] = sigma * TRADING_DAYS

    diag = pd.Series(np.diag(cov.to_numpy()), index=cov.index)
    fallback = float(diag.median()) if diag.notna().any() else 0.30**2
    for name in returns.columns.difference(good):
        cov.loc[name, :] = 0.0
        cov.loc[:, name] = 0.0
        cov.loc[name, name] = fallback
    return cov.astype(float)
