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
    * If the traded turnover ``Σ|Δw|`` exceeds ``max_turnover``:
      - full exits (names no longer in the target) are always executed, so
        deteriorated positions never linger as small residual holdings;
      - all other trades move a fraction ``k`` of the way to target, with
        ``k`` chosen so that the total stays within the cap when possible;
      - exit proceeds not used by those trades are reinvested in names still
        below target (never above it), so the cap does not create cash drag.
      When exits alone exceed the budget the cap is exceeded by design.
    * The initial funding of an empty portfolio is exempt from the cap.
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
        executed = _cap_turnover(cur, tgt, executed, is_exit, max_turnover)
        turnover = float(executed.abs().sum())
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


def _cap_turnover(
    cur: pd.Series,
    tgt: pd.Series,
    executed: pd.Series,
    is_exit: pd.Series,
    max_turnover: float,
) -> pd.Series:
    exits = executed.where(is_exit, 0.0)
    rest = executed.where(~is_exit, 0.0)
    exit_turnover = float(exits.abs().sum())
    rest_turnover = float(rest.abs().sum())

    # Exit proceeds are reinvested, so each unit sold costs ~2 units of turnover
    # unless the scaled trades already absorb it.
    budget = max_turnover - 2.0 * exit_turnover
    denom = rest_turnover - exit_turnover
    k = min(max(budget / denom, 0.0), 1.0) if denom > 0 else 0.0
    final = cur + exits + k * rest

    intended = float((cur + executed).sum())
    shortfall = intended - float(final.sum())
    gap = (tgt - final).where(~is_exit & (tgt > 0), 0.0).clip(lower=0.0)
    if shortfall > 1e-12 and gap.sum() > 0:
        final = final + gap * min(shortfall / float(gap.sum()), 1.0)
    return final - cur
