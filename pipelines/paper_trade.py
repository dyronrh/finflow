"""Paper trading (Sprint 8): propose → approve → execute → sync → reconcile.

    python pipelines/paper_trade.py status
    python pipelines/paper_trade.py propose --by analyst
    python pipelines/paper_trade.py approve rb_2026-10-05_ab12cd34 --by "nombre" --role approver
    python pipelines/paper_trade.py execute rb_2026-10-05_ab12cd34 --by "nombre"
    python pipelines/paper_trade.py sync | reconcile | history [proposal_id]
    python pipelines/paper_trade.py kill --reason "..." --by "nombre"   /   unkill --by "nombre"

Brokers: --broker sim (local simulation, default) or --broker alpaca
(Alpaca *paper* account; set ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY).
Execution additionally requires GLOBAL_TRADING_ENABLED=true (env or .env).
Live trading is not supported.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from datetime import date
from pathlib import Path

from _common import ROOT, add_source_args, load_source

sys.path.insert(0, str(ROOT))

from apps.api.services.research import ResearchService
from paper_trading.broker import AlpacaPaperBroker, SimulatedBroker
from paper_trading.controls import (
    DEFAULT_KILL_FILE,
    TradingControls,
    TradingDisabled,
    activate_kill_switch,
    deactivate_kill_switch,
)
from paper_trading.desk import PaperTradingDesk, PreTradeLimits, WorkflowError
from paper_trading.ledger import Ledger
from quant_core.config import default_strategy_config, load_strategy_config
from quant_core.execution.costs import TransactionCostModel

PAPER_DIR = ROOT / "data" / "local_dev_only" / "paper"


def build_desk(args) -> PaperTradingDesk:
    config = load_strategy_config(args.config) if args.config else default_strategy_config()
    market = load_source(args, "2016-01-01", date.today().isoformat())
    research = ResearchService(market, config)
    traded = market.price_close if market.price_close is not None else market.close

    def last_prices(symbols: list[str]) -> dict[str, float]:
        out = {}
        for s in symbols:
            if s in traded.columns:
                series = traded[s].dropna()
                if len(series):
                    out[s] = float(series.iloc[-1])
        return out

    if args.broker == "alpaca":
        broker = AlpacaPaperBroker()
    else:
        broker = SimulatedBroker(
            PAPER_DIR / f"sim_broker_{args.source}.json",
            last_prices,
            initial_cash=args.initial_cash,
            costs=TransactionCostModel.from_config(config.costs),
        )
    ledger = Ledger(PAPER_DIR / f"ledger_{args.source}_{broker.name}.sqlite")
    limits = PreTradeLimits(require_four_eyes=args.four_eyes)
    return PaperTradingDesk(broker, ledger, research, TradingControls.from_env(), limits)


def show(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "action",
        choices=[
            "status",
            "propose",
            "approve",
            "reject",
            "execute",
            "sync",
            "reconcile",
            "history",
            "kill",
            "unkill",
        ],
    )
    parser.add_argument("proposal_id", nargs="?")
    parser.add_argument("--by", default=None, help="who performs the action (audited)")
    parser.add_argument(
        "--role", default="analyst", help="viewer|analyst|portfolio_manager|approver|admin"
    )
    parser.add_argument("--reason", default=None)
    parser.add_argument("--broker", choices=["sim", "alpaca"], default="sim")
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument(
        "--four-eyes", action="store_true", help="approver must differ from proposer"
    )
    parser.add_argument("--config", type=Path)
    add_source_args(parser)
    args = parser.parse_args()
    who = args.by or getpass.getuser()

    if args.action == "kill":
        path = activate_kill_switch(args.reason or "manual", who)
        print(f"KILL SWITCH ACTIVE ({path}). No order will be sent until `unkill`.")
        return
    if args.action == "unkill":
        print("kill switch released" if deactivate_kill_switch() else "kill switch was not active")
        return

    desk = build_desk(args)
    try:
        if args.action == "status":
            show(desk.status())
        elif args.action == "propose":
            pid = desk.propose(who)
            p = desk.ledger.proposal(pid)["payload"]
            print(
                f"\nproposal {pid}: PENDING_APPROVAL · {len(p['orders'])} orders · "
                f"turnover {p['estimated_turnover']:.1%} · est. cost ${p['estimated_cost_usd']:.2f}"
            )
            for o in p["orders"]:
                print(
                    f"  {o['side']:<4} {o['security_id']:<8} qty {o['qty']:>10.4f} "
                    f"~${o['est_notional']:>10,.2f}  "
                    f"{o['current_weight']:.1%} → {o['target_weight']:.1%}"
                    f"  {'; '.join(o['reason'])}"
                )
            print(f"risk before: {p['risk_before']}\nrisk after:  {p['risk_after']}")
            if p["risk_actions"]:
                print(f"risk actions: {p['risk_actions']}")
            for a in p["alerts"]:
                print(f"  [{a['severity']}] {a['message']}")
            print(f"pre-trade checks: {'OK' if p['pre_trade_ok'] else p['pre_trade_checks']}")
            if p["unmanaged_positions"]:
                print(f"unmanaged positions (left untouched): {p['unmanaged_positions']}")
            print(f"\nnext: approve {pid} --by <name> --role approver")
        elif args.action in {"approve", "reject", "execute", "history"} and not args.proposal_id:
            raise SystemExit("proposal_id is required")
        elif args.action == "approve":
            desk.approve(args.proposal_id, who, args.role, args.reason)
            print(f"{args.proposal_id} APPROVED by {who}")
        elif args.action == "reject":
            desk.reject(args.proposal_id, who, args.reason or "rejected")
            print(f"{args.proposal_id} REJECTED by {who}")
        elif args.action == "execute":
            show(desk.execute(args.proposal_id, who))
        elif args.action == "sync":
            print(f"orders updated: {desk.sync(who)}")
        elif args.action == "reconcile":
            breaks = desk.reconcile(who)
            print("reconciled: no breaks" if not breaks else "BREAKS:")
            for b in breaks:
                print(f"  {b}")
        elif args.action == "history":
            show(desk.ledger.audit_trail(args.proposal_id))
    except (WorkflowError, PermissionError, TradingDisabled) as exc:
        raise SystemExit(f"BLOCKED: {exc}") from None
    print(f"\n(kill switch file: {DEFAULT_KILL_FILE})" if args.action == "status" else "")


if __name__ == "__main__":
    main()
