"""Transaction-cost model (README §11.3)."""

from __future__ import annotations

from dataclasses import dataclass

from quant_core.config import CostConfig


@dataclass(frozen=True)
class TransactionCostModel:
    commission_bps: float = 1.0
    half_spread_bps: float = 2.0
    slippage_bps: float = 5.0

    @classmethod
    def from_config(cls, config: CostConfig) -> TransactionCostModel:
        return cls(config.commission_bps, config.half_spread_bps, config.slippage_bps)

    @property
    def total_bps(self) -> float:
        return self.commission_bps + self.half_spread_bps + self.slippage_bps

    def cost(self, notional: float) -> float:
        """Conservative one-way cost in USD for a trade of ``notional`` USD."""
        return abs(notional) * self.total_bps / 10_000

    def breakdown(self, notional: float) -> dict[str, float]:
        n = abs(notional)
        return {
            "commission_cost": n * self.commission_bps / 10_000,
            "bid_ask_spread_cost": n * self.half_spread_bps / 10_000,
            "slippage_cost": n * self.slippage_bps / 10_000,
        }
