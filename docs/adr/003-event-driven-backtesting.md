# ADR 003 — Event-driven backtesting with t+1 fills

**Status:** accepted · **Date:** 2026-10-05

## Decision
* Signals are computed with data up to the close of day `t`.
* Orders fill at the **open of `t+1`**; costs (commission + half-spread +
  slippage, in bps) are charged on every fill and trade sizes are net of costs.
* Rebalances run on the last trading day of each month (or half-month) and go
  through the same no-trade band and turnover cap as live proposals
  (`quant_core.portfolio.rebalance.plan_rebalance`).
* Every run records strategy version, config hash, data version, cost
  assumptions, seed and git commit.

## Consequences
`tests/integration/test_backtest.py` asserts `fill_date > signal_date`, fills at the
open price, determinism, and a shared calendar with the benchmark.
Walk-forward evaluation and market-impact modelling are not implemented yet.
