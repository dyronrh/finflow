# ADR 002 — Point-in-time data

**Status:** accepted · **Date:** 2026-10-05

## Context
Backtests that use restated or late-published data, or today's index members,
overstate performance (README §2.1).

## Decision
* Every non-price record carries `event_time`, `available_at` and `ingested_at`.
* Feature snapshots read only `available_at <= as_of` (`data_platform.point_in_time`)
  and prices up to the close of `as_of`.
* `tests/unit/test_point_in_time.py` mutates all data after `as_of` and asserts
  the snapshot is unchanged.

## Consequences
* Fundamentals enter features only after their publication lag (25–60 days in
  the synthetic data).
* Still open: point-in-time universe membership, delistings and restatements
  need a real vendor feed.
