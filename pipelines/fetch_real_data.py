"""Download real market data (Yahoo Finance prices + SEC EDGAR fundamentals).

    export SEC_USER_AGENT='finflow research you@example.com'
    uv run python pipelines/fetch_real_data.py --start 2009-01-01

Raw responses are cached under data/local_dev_only/real/raw and the assembled
dataset under data/local_dev_only/real/processed; reruns reuse the SEC cache.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

from _common import DEFAULT_CACHE, sec_user_agent
from data_platform.real_market import fetch_real_market


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--start", default="2009-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--max-securities", type=int, help="limit universe (for a quick trial)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    market = fetch_real_market(
        args.start, args.end, sec_user_agent(), args.cache, max_securities=args.max_securities
    )
    f = market.fundamentals
    print(f"data_version: {market.data_version}")
    print(f"securities with prices + SEC facts: {len(market.securities)}")
    print(f"price history: {market.dates[0].date()} → {market.dates[-1].date()}")
    print(
        f"fundamental rows: {len(f)}; median filing lag: "
        f"{(f['available_at'] - f['event_time']).dt.days.median():.0f} days"
    )
    print(f"without prices (likely delisted): {len(market.metadata['tickers_without_prices'])}")
    print(f"without SEC facts: {len(market.metadata['tickers_without_sec_facts'])}")


if __name__ == "__main__":
    main()
