"""Shared CLI helpers for pipelines."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))

from data_platform.market import MarketData  # noqa: E402
from data_platform.real_market import DEFAULT_CACHE, load_market  # noqa: E402
from data_platform.synthetic import generate_synthetic_market  # noqa: E402


def add_source_args(parser) -> None:
    parser.add_argument(
        "--source",
        choices=["real", "synthetic"],
        default="real",
        help="real = cached Yahoo Finance + SEC EDGAR data (run fetch_real_data.py first)",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE / "processed")
    parser.add_argument("--seed", type=int, default=42, help="synthetic market seed")


def load_source(args, start: str, end: str) -> MarketData:
    if args.source == "synthetic":
        first = f"{int(start[:4]) - 2}-01-01"
        return generate_synthetic_market(start=first, end=end, seed=args.seed)
    return load_market(args.cache)


def sec_user_agent() -> str:
    agent = os.environ.get("SEC_USER_AGENT", "")
    if not agent:
        raise SystemExit(
            "Set SEC_USER_AGENT with a contact e-mail, e.g.\n"
            "  export SEC_USER_AGENT='finflow research you@example.com'\n"
            "(required by the SEC fair-access policy)."
        )
    return agent
