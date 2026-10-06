"""Show the strategy's decision for specific tickers on a given date.

    uv run --extra real-data python pipelines/score_universe.py --tickers TSLA NVDA
    uv run --extra real-data python pipelines/score_universe.py --top 20

Research output only: it is not investment advice, and the strategy has not
shown an out-of-sample edge (see tune_strategy.py).
"""

from __future__ import annotations

import argparse
import sys

from _common import ROOT, add_source_args, load_source

sys.path.insert(0, str(ROOT))

from apps.api.services.research import ResearchService, UnknownSecurity
from quant_core.config import default_strategy_config, load_strategy_config
from quant_core.portfolio.construction import build_target_weights

FAMILIES = ("value", "growth", "profitability", "momentum", "revisions")


def _fmt(value: object, pattern: str = "{:.1f}") -> str:
    return "n/a" if value is None else pattern.format(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--tickers", nargs="*", default=[], help="e.g. TSLA NVDA")
    parser.add_argument("--top", type=int, default=0, help="also list the N best-ranked names")
    parser.add_argument("--as-of", default=None, help="decision date (default: last available)")
    parser.add_argument("--config", help="strategy YAML (default: v0.1.0)")
    add_source_args(parser)
    args = parser.parse_args()

    config = load_strategy_config(args.config) if args.config else default_strategy_config()
    market = load_source(args, "2016-01-01", "2026-12-31")
    service = ResearchService(market, config)
    as_of = service.resolve_as_of(args.as_of)
    signals = service.signals(as_of)
    model_portfolio = build_target_weights(signals, config.portfolio)

    print(f"\nas of {as_of.date()} · strategy {config.strategy_version}", end="")
    print(f" · {len(signals)} eligible names")
    known = set(market.securities["security_id"])
    for raw in args.tickers:
        sid = raw.strip().upper().replace(".", "-")
        print(f"\n=== {sid} ===")
        if sid not in known:
            print(
                "  not in the dataset: not an S&P 500 member in the downloaded window, "
                "no Yahoo prices / SEC filings, or not publicly traded (e.g. private companies)."
            )
            continue
        try:
            a = service.security_analysis(sid, as_of)
        except UnknownSecurity:
            print("  no data on this date.")
            continue
        if not a["eligible"] or a["decision"] is None:
            reasons = ", ".join(a["ineligibility_reasons"]) or "insufficient fundamental coverage"
            print(f"  NOT SCORED — {reasons}")
            continue
        weight = model_portfolio.get(sid)
        print(f"  decision:   {a['decision']}")
        print(
            f"  composite:  {_fmt(a['composite_score'])}/100 "
            f"(percentile {_fmt(a['composite_percentile'], '{:.0%}')} of eligible universe)"
        )
        scores = a["factor_scores"] or {}
        sector = a["sector_median_scores"] or {}
        for f in FAMILIES:
            print(f"    {f:<14}{_fmt(scores.get(f)):>6}   sector median {_fmt(sector.get(f)):>6}")
        perf = a["performance"]
        print(
            "  returns:    "
            + "  ".join(
                f"{k.removeprefix('return_')} {_fmt(v, '{:+.0%}')}" for k, v in perf.items()
            )
        )
        print(f"  risk flags: {', '.join(a['risk_flags']) or 'none'}")
        print(
            "  model portfolio weight: " + (f"{weight:.1%}" if weight is not None else "not held")
        )
        for line in a["explanation"]:
            print(f"   - {line}")

    if args.top:
        print(f"\n=== Top {args.top} by composite score ===")
        cols = ["ticker", "sector_id", "decision", "composite_score", "composite_percentile"]
        print(signals[cols].head(args.top).round(3).to_string(index=False))

    print(
        "\nResearch output, not investment advice. The current strategy has not shown "
        "an out-of-sample edge over its benchmark; do not trade on these labels."
    )


if __name__ == "__main__":
    main()
