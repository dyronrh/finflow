"""Approximate SIC → GICS-like sector mapping (GICS itself is licensed).

Used only for securities without a GICS sector (e.g. removed index members).
"""

from __future__ import annotations

# (low, high, sector) — first match wins, so specific ranges come first.
_RANGES: tuple[tuple[int, int, str], ...] = (
    (1311, 1389, "energy"),
    (2830, 2836, "health_care"),
    (2840, 2844, "consumer_staples"),
    (2900, 2999, "energy"),
    (3570, 3579, "information_technology"),
    (3630, 3639, "consumer_discretionary"),
    (3711, 3716, "consumer_discretionary"),
    (3840, 3851, "health_care"),
    (5122, 5122, "health_care"),
    (5140, 5149, "consumer_staples"),
    (5400, 5499, "consumer_staples"),
    (5912, 5912, "consumer_staples"),
    (6500, 6599, "real_estate"),
    (6798, 6798, "real_estate"),
    (7011, 7011, "consumer_discretionary"),
    (7370, 7379, "information_technology"),
    (7800, 7899, "communication_services"),
    (100, 999, "consumer_staples"),
    (1000, 1499, "materials"),
    (1500, 1799, "industrials"),
    (2000, 2199, "consumer_staples"),
    (2200, 2399, "consumer_discretionary"),
    (2700, 2799, "communication_services"),
    (2400, 2699, "materials"),
    (2800, 2899, "materials"),
    (3100, 3199, "consumer_discretionary"),
    (3000, 3399, "materials"),
    (3400, 3599, "industrials"),
    (3600, 3699, "information_technology"),
    (3700, 3799, "industrials"),
    (3800, 3899, "information_technology"),
    (3900, 3999, "consumer_discretionary"),
    (4000, 4799, "industrials"),
    (4800, 4899, "communication_services"),
    (4900, 4999, "utilities"),
    (5000, 5199, "industrials"),
    (5200, 5999, "consumer_discretionary"),
    (6000, 6799, "financials"),
    (8000, 8099, "health_care"),
    (7000, 8999, "industrials"),
)


def sic_to_sector(sic: int | str | None) -> str | None:
    try:
        code = int(sic)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    for low, high, sector in _RANGES:
        if low <= code <= high:
            return sector
    return None
