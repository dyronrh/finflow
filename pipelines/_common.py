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
    market = load_market(args.cache)
    print(f"data_version: {market.data_version}")
    print(f"universe: {market.metadata.get('universe', 'unknown')}")
    for message in market.metadata.get("warnings", []):
        print(f"WARNING: {message}")
    return market


def _from_dotenv(name: str) -> str:
    """Read ``name`` from the repo's .env (gitignored) without extra dependencies."""
    path = ROOT / ".env"
    if not path.exists():
        return ""
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == name:
            return value.strip().strip("'\"")
    return ""


def sec_user_agent() -> str:
    agent = os.environ.get("SEC_USER_AGENT", "") or _from_dotenv("SEC_USER_AGENT")
    if not agent:
        raise SystemExit(
            "Set SEC_USER_AGENT with a contact e-mail (required by the SEC fair-access\n"
            "policy), either for this shell:\n"
            "  export SEC_USER_AGENT='finflow research you@example.com'\n"
            "or once and for all in the repo's .env file (not committed):\n"
            "  echo \"SEC_USER_AGENT='finflow research you@example.com'\" >> .env"
        )
    return agent
