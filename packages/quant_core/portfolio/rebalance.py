"""Rebalance proposals with no-trade bands and turnover limits (README §11)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RebalancePlan:
    orders: pd.DataFrame
    """Columns: security_id, side, current_weight, target_weight, delta_weight,
    final_weight, reason."""
    turnover: float
    turnover_capped: bool

    @property
    def final_weights(self) -> pd.Series:
        final = self.orders.set_index("security_id")["final_weight"]
        return final[final > 1e-12]


def plan_rebalance(
    current: pd.Series,
    target: pd.Series,
    trade_band: float,
    max_turnover: float,
) -> RebalancePlan:
    """Turn target weights into trades.

    * Changes smaller than ``trade_band`` are skipped (except full exits and
      new entries, which always trade).
    * If the traded turnover ``Σ|Δw|`` exceeds ``max_turnover`` all trades are
      scaled down proportionally. The initial funding of an empty portfolio is
      exempt from the turnover cap.
    """
    index = current.index.union(target.index)
    cur = current.reindex(index, fill_value=0.0).astype(float)
    tgt = target.reindex(index, fill_value=0.0).astype(float)
    delta = tgt - cur

    is_exit = (tgt <= 0) & (cur > 0)
    is_entry = (cur <= 0) & (tgt > 0)
    trade = (delta.abs() > trade_band) | is_exit | is_entry
    executed = delta.where(trade, 0.0)

    turnover = float(executed.abs().sum())
    initial_funding = float(cur.abs().sum()) == 0.0
    capped = False
    if not initial_funding and turnover > max_turnover > 0:
        executed = executed * (max_turnover / turnover)
        turnover = max_turnover
        capped = True

    reasons = pd.Series("", index=index)
    reasons[is_entry] = "NEW_POSITION"
    reasons[is_exit] = "EXIT_POSITION"
    reasons[~is_entry & ~is_exit & trade & (delta > 0)] = "INCREASE_TO_TARGET"
    reasons[~is_entry & ~is_exit & trade & (delta < 0)] = "DECREASE_TO_TARGET"
    reasons[~trade & (delta != 0)] = "WITHIN_TRADE_BAND"

    side = pd.Series("HOLD", index=index)
    side[executed > 0] = "BUY"
    side[executed < 0] = "SELL"

    orders = pd.DataFrame(
        {
            "security_id": index,
            "side": side.to_numpy(),
            "current_weight": cur.to_numpy(),
            "target_weight": tgt.to_numpy(),
            "delta_weight": executed.to_numpy(),
            "final_weight": (cur + executed).clip(lower=0.0).to_numpy(),
            "reason": reasons.to_numpy(),
        }
    )
    return RebalancePlan(orders=orders, turnover=turnover, turnover_capped=capped)
