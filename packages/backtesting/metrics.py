"""Performance and risk metrics (README §12.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def cvar(returns: pd.Series, level: float = 0.95) -> float:
    """Expected shortfall of daily returns, reported as a negative number."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    cutoff = r.quantile(1.0 - level)
    return float(r[r <= cutoff].mean())


def performance_summary(equity: pd.Series, risk_free_rate: float = 0.0) -> dict[str, float]:
    equity = equity.dropna()
    returns = equity.pct_change().dropna()
    if len(returns) < 2:
        return {}
    years = len(returns) / TRADING_DAYS
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0)
    vol = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    excess = returns - risk_free_rate / TRADING_DAYS
    downside = excess[excess < 0]
    downside_dev = float(np.sqrt((downside**2).sum() / len(excess)) * np.sqrt(TRADING_DAYS))
    mdd = max_drawdown(equity)
    annual_excess = float(excess.mean() * TRADING_DAYS)
    return {
        "cumulative_return": total,
        "cagr": cagr,
        "annualized_volatility": vol,
        "sharpe_ratio": annual_excess / vol if vol > 0 else float("nan"),
        "sortino_ratio": annual_excess / downside_dev if downside_dev > 0 else float("nan"),
        "max_drawdown": mdd,
        "calmar_ratio": cagr / abs(mdd) if mdd < 0 else float("nan"),
        "cvar_95_daily": cvar(returns, 0.95),
        "downside_deviation": downside_dev,
        "hit_rate": float((returns > 0).mean()),
    }


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    joined = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(joined) < 2:
        return float("nan")
    cov = np.cov(joined.iloc[:, 0], joined.iloc[:, 1], ddof=1)
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else float("nan")
