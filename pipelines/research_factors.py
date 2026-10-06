"""Evaluate candidate factors with a research sample and an untouched holdout.

    uv run --extra real-data python pipelines/research_factors.py --source real
    uv run python pipelines/research_factors.py --source synthetic

Outputs (reports/factors_<source>_<timestamp>/):
  verdicts.csv       one row per candidate: IC, Newey-West t, BH q-value, holdout, verdict
  research.csv       full statistics on the research sample
  holdout.csv        same statistics on the holdout (reported, never used to select)
  correlations.csv   average cross-sectional correlation between candidates
  ic_by_year.csv     mean 1-month IC per candidate and calendar year
  report.json        metadata and the composite of research-passing factors

Verdicts: ACCEPT (significant after BH, right sign, holdout IC > 0 with t >= 1),
FAILS_HOLDOUT, INVERTED (predicts against the hypothesis: never flipped
silently), NOT_SIGNIFICANT. An ACCEPT is a candidate for a new strategy
version, which still needs a backtest and human review (README §8.3).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from _common import add_source_args, load_source
from quant_core.config import default_strategy_config, load_strategy_config
from research.factors import pairs_above, research_factors


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--start", default=None, help="default: 2012-01-01 real, 2016-01-01 synthetic"
    )
    parser.add_argument("--end", default=datetime.now().date().isoformat())
    parser.add_argument("--holdout-start", default=None, help="default: 3 years before --end")
    parser.add_argument("--fdr", type=float, default=0.10, help="Benjamini-Hochberg level")
    parser.add_argument("--config", type=Path, help="strategy YAML for eligibility rules")
    parser.add_argument("--out", type=Path, default=Path("reports"))
    add_source_args(parser)
    args = parser.parse_args()

    start = args.start or ("2012-01-01" if args.source == "real" else "2016-01-01")
    holdout = args.holdout_start or f"{int(args.end[:4]) - 3}{args.end[4:]}"
    config = load_strategy_config(args.config) if args.config else default_strategy_config()
    market = load_source(args, start, args.end)

    report = research_factors(
        market, config, start, args.end, holdout, fdr=args.fdr, progress=print
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out / f"factors_{args.source}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    report.verdicts.to_csv(run_dir / "verdicts.csv")
    report.research.to_csv(run_dir / "research.csv")
    report.holdout.to_csv(run_dir / "holdout.csv")
    report.correlations.to_csv(run_dir / "correlations.csv")
    report.ic_yearly.to_csv(run_dir / "ic_by_year.csv")
    redundant = pairs_above(report.correlations)
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "metadata": report.metadata,
                "accepted": report.accepted,
                "composite": report.composite,
                "redundant_pairs": redundant,
            },
            indent=2,
            default=str,
        )
    )

    cols = [
        "family",
        "mean_ic_1m",
        "nw_t_1m",
        "q_value_bh",
        "holdout_mean_ic_1m",
        "holdout_nw_t_1m",
        "verdict",
    ]
    extra = ["q5_q1_annual", "monotonicity", "top_quintile_turnover", "coverage"]
    table = report.verdicts[cols].join(report.research[extra])
    print(
        f"\nresearch {start} → {holdout}, holdout {holdout} → {args.end}, "
        f"{report.metadata['months']} months, "
        f"{report.metadata['n_candidates_tested']} candidates tested"
    )
    print(table.to_string(float_format=lambda v: f"{v:+.3f}"))
    print(f"\naccepted: {', '.join(report.accepted) or 'none'}")
    for label, stats in report.composite.items():
        print(
            f"composite ({label}): mean IC 1m {stats.get('mean_ic_1m', float('nan')):+.4f}  "
            f"NW t {stats.get('nw_t_1m', float('nan')):+.2f}  "
            f"Q5-Q1 {stats.get('q5_q1_annual', float('nan')):+.2%}/yr"
        )
    if redundant:
        print(
            "redundant pairs (|corr| >= 0.7): "
            + ", ".join(f"{a}~{b} {c:+.2f}" for a, b, c in redundant[:8])
        )
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
