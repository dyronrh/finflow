"""Inspect and approve registered models (README §13.3).

uv run python pipelines/model_registry.py list
uv run python pipelines/model_registry.py show ranker_gbm v001
uv run python pipelines/model_registry.py approve ranker_gbm v001 --by "nombre"
uv run python pipelines/model_registry.py reject|retire ranker_gbm v001 --by "nombre"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import _common  # noqa: F401  (adds packages/ to sys.path)
from ml.registry import DEFAULT_ROOT, ModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("action", choices=["list", "show", "approve", "reject", "retire"])
    parser.add_argument("name", nargs="?")
    parser.add_argument("version", nargs="?")
    parser.add_argument("--by", help="who approves/rejects (recorded)")
    parser.add_argument("--registry", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    reg = ModelRegistry(args.registry)

    if args.action == "list":
        for m in reg.list():
            ic = m.metrics.get("ic", {}).get("holdout", {}).get("ml", {}).get("mean_ic")
            print(
                f"{m.model_name:<14}{m.model_version:<6}{m.approval_status:<16}"
                f"trained→{m.train_end_date}  holdout IC={ic if ic is None else round(ic, 4)}"
                f"  drift={m.metrics.get('drift_overall')}"
            )
        return
    if not (args.name and args.version):
        raise SystemExit("name and version are required")
    if args.action == "show":
        print(json.dumps(asdict(reg.metadata(args.name, args.version)), indent=2, default=str))
        return
    if not args.by:
        raise SystemExit("--by is required: approvals are audited")
    status = {"approve": "APPROVED", "reject": "REJECTED", "retire": "RETIRED"}[args.action]
    meta = reg.set_status(args.name, args.version, status, args.by)
    print(f"{meta.model_name} {meta.model_version} → {meta.approval_status} by {meta.approved_by}")


if __name__ == "__main__":
    main()
