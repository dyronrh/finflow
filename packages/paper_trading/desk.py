"""Paper-trading workflow: propose → approve → execute → sync → reconcile (Sprint 8).

Hard rules (README §2.2, §14.3, §15):
* proposals never send orders;
* only a user acting with the ``approver`` role can approve, and only proposals
  whose pre-trade checks passed;
* execution requires an approved, non-expired proposal and passing trading
  controls (kill switch checked before every single order);
* a proposal executes at most once;
* every step is written to the ledger's audit log.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from apps.api.services.research import ResearchService
from paper_trading.broker import Broker
from paper_trading.controls import TradingControls, TradingDisabled
from paper_trading.ledger import Ledger
from quant_core.execution.costs import TransactionCostModel
from quant_core.risk.alerts import drawdown_alert

ROLES = ("viewer", "analyst", "portfolio_manager", "approver", "admin")


class WorkflowError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreTradeLimits:
    max_order_pct_nav: float = 0.10
    max_orders: int = 80
    min_order_usd: float = 25.0
    cash_buffer_pct: float = 0.005
    """Share of equity kept in cash for slippage and commissions."""
    max_data_staleness_bdays: int = 5
    proposal_ttl_hours: float = 72.0
    approval_ttl_hours: float = 24.0
    require_four_eyes: bool = False
    """If true, the approver must differ from the proposer."""


def _age_hours(iso: str, now: datetime) -> float:
    return (now - datetime.fromisoformat(iso)).total_seconds() / 3600.0


class PaperTradingDesk:
    def __init__(
        self,
        broker: Broker,
        ledger: Ledger,
        research: ResearchService,
        controls: TradingControls,
        limits: PreTradeLimits | None = None,
        clock=lambda: datetime.now(UTC),
    ) -> None:
        self.broker = broker
        self.ledger = ledger
        self.research = research
        self.controls = controls
        self.limits = limits or PreTradeLimits()
        self.clock = clock
        self.ledger.clock = clock  # one clock for timestamps and expiry checks
        self.costs = TransactionCostModel.from_config(research.config.costs)

    # ------------------------------------------------------------------ helpers
    def _data_age_bdays(self) -> int:
        last = self.research.market.dates[-1].date()
        today = self.clock().date()
        return int(np.busday_count(last, today)) if today > last else 0

    def current_book(
        self,
    ) -> tuple[pd.Series, dict[str, float], dict[str, float], dict[str, float]]:
        """(weights of managed names, account, all positions, prices)."""
        account = self.broker.account()
        positions = self.broker.positions()
        prices = self.broker.last_prices(list(positions)) if positions else {}
        equity = account["equity"]
        known = set(self.research.market.securities["security_id"])
        weights = pd.Series(
            {s: q * prices.get(s, 0.0) / equity for s, q in positions.items() if s in known},
            dtype=float,
        )
        return weights, account, positions, prices

    # ------------------------------------------------------------------ propose
    def propose(self, by: str) -> str:
        age = self._data_age_bdays()
        if age > self.limits.max_data_staleness_bdays:
            raise WorkflowError(
                f"market data is {age} business days old (max "
                f"{self.limits.max_data_staleness_bdays}); run `make fetch-data` first"
            )
        weights, account, positions, _ = self.current_book()
        unmanaged = sorted(set(positions) - set(weights.index))
        as_of = self.research.resolve_as_of(None)
        payload = self.research.rebalance_proposal(as_of, weights.to_dict(), account["equity"])
        prices = self.broker.last_prices([o["security_id"] for o in payload["orders"]])

        orders, checks = [], []
        held = positions
        for order in payload["orders"]:
            if order["side"] == "HOLD":
                continue
            sid = order["security_id"]
            price = prices.get(sid)
            notional = order["delta_weight"] * account["equity"]
            if not price:
                checks.append({"check": "PRICE_AVAILABLE", "ok": False, "security_id": sid})
                continue
            qty = round(abs(notional) / price, 4)
            if order["side"] == "BUY":
                qty = round(qty * (1 - self.limits.cash_buffer_pct), 4)
            if order["side"] == "SELL":
                qty = min(qty, held.get(sid, 0.0))
                if order["target_weight"] <= 0:
                    qty = held.get(sid, 0.0)  # exits close the full position
            if qty * price < self.limits.min_order_usd:
                continue
            is_exit = order["side"] == "SELL" and order["target_weight"] <= 0
            too_big = qty * price > self.limits.max_order_pct_nav * account["equity"] + 1e-6
            if too_big and not is_exit:
                checks.append(
                    {
                        "check": "MAX_ORDER_SIZE",
                        "ok": False,
                        "security_id": sid,
                        "notional": qty * price,
                    }
                )
            orders.append({**order, "qty": qty, "est_price": price, "est_notional": qty * price})
        if len(orders) > self.limits.max_orders:
            checks.append({"check": "MAX_ORDERS", "ok": False, "count": len(orders)})
        payload["orders"] = orders
        payload["unmanaged_positions"] = unmanaged
        payload["pre_trade_checks"] = checks
        payload["pre_trade_ok"] = all(c["ok"] for c in checks)
        payload["broker_equity"] = account["equity"]
        pid = self.ledger.add_proposal(payload, by, self.broker.name)
        self.ledger.add_snapshot("broker", account["cash"], account["equity"], positions)
        return pid

    # ------------------------------------------------------------------ approve
    def approve(self, pid: str, by: str, role: str, reason: str | None = None) -> None:
        if role != "approver":
            self.ledger.audit(by, "APPROVAL_DENIED", {"role": role}, pid)
            raise PermissionError(f"role {role!r} cannot approve; 'approver' required")
        p = self.ledger.proposal(pid)
        if p["status"] != "PENDING_APPROVAL":
            raise WorkflowError(f"proposal is {p['status']}")
        if _age_hours(p["created_at"], self.clock()) > self.limits.proposal_ttl_hours:
            self.ledger.set_proposal_status(pid, "EXPIRED", by)
            raise WorkflowError("proposal expired; create a new one")
        if not p["payload"].get("pre_trade_ok", False):
            raise WorkflowError(
                f"pre-trade checks failed: {p['payload']['pre_trade_checks']}; cannot approve"
            )
        if self.limits.require_four_eyes and by == p["created_by"]:
            raise PermissionError("four-eyes rule: approver must differ from proposer")
        self.ledger.add_approval(pid, by, role, "APPROVED", reason)
        self.ledger.set_proposal_status(pid, "APPROVED", by, {"reason": reason})

    def reject(self, pid: str, by: str, reason: str) -> None:
        p = self.ledger.proposal(pid)
        if p["status"] not in {"PENDING_APPROVAL", "APPROVED"}:
            raise WorkflowError(f"proposal is {p['status']}")
        self.ledger.add_approval(pid, by, "approver", "REJECTED", reason)
        self.ledger.set_proposal_status(pid, "REJECTED", by, {"reason": reason})

    # ------------------------------------------------------------------ execute
    def execute(self, pid: str, by: str) -> dict[str, int]:
        try:
            self.controls.assert_can_trade("paper")
        except TradingDisabled as exc:
            self.ledger.audit(by, "EXECUTION_BLOCKED", {"reason": str(exc)}, pid)
            raise
        p = self.ledger.proposal(pid)
        if p["status"] != "APPROVED":
            raise WorkflowError(f"proposal is {p['status']}; only APPROVED proposals execute")
        if self.ledger.orders(pid):
            raise WorkflowError("proposal already has orders; refusing to execute twice")
        approval = self.ledger.approval(pid)
        if (
            approval is None
            or _age_hours(approval["decided_at"], self.clock()) > self.limits.approval_ttl_hours
        ):
            self.ledger.set_proposal_status(pid, "EXPIRED", by)
            raise WorkflowError("approval expired; propose and approve again")
        if self._data_age_bdays() > self.limits.max_data_staleness_bdays:
            raise WorkflowError("market data became stale since the proposal")

        self.ledger.set_proposal_status(pid, "EXECUTING", by)
        orders = sorted(p["payload"]["orders"], key=lambda o: o["side"] != "SELL")  # sells first
        counts = {"filled": 0, "submitted": 0, "rejected": 0, "skipped": 0}
        for o in orders:
            if self.controls.kill_switch_active:  # can stop mid-run
                self.ledger.audit(by, "EXECUTION_HALTED_BY_KILL_SWITCH", {}, pid)
                counts["skipped"] += 1
                continue
            qty = self._affordable_qty(o)
            if qty <= 0:
                self.ledger.audit(
                    by, "ORDER_SKIPPED_NO_CASH", {"security_id": o["security_id"]}, pid
                )
                counts["skipped"] += 1
                continue
            oid = self.ledger.add_order(pid, o["security_id"], o["side"], qty, o["est_price"])
            result = self.broker.submit_market_order(o["security_id"], o["side"], qty, oid)
            self._record(oid, o["security_id"], o["side"], result)
            key = {"FILLED": "filled", "REJECTED": "rejected"}.get(result.status, "submitted")
            counts[key] += 1
        status = (
            "EXECUTED"
            if counts["filled"] == len(orders)
            else "FAILED"
            if counts["filled"] + counts["submitted"] == 0 and orders
            else "PARTIALLY_EXECUTED"
        )
        self.ledger.set_proposal_status(pid, status, by, counts)
        return counts

    def _affordable_qty(self, order: dict) -> float:
        """Buys are capped by the cash actually available (prices move after proposing)."""
        if order["side"] != "BUY":
            return order["qty"]
        cash = self.broker.account()["cash"]
        price = self.broker.last_prices([order["security_id"]]).get(order["security_id"], 0.0)
        if price <= 0:
            return 0.0
        unit = price * (1 + self.costs.total_bps / 10_000)
        return round(min(order["qty"], max(cash, 0.0) / unit * 0.999), 4)

    def _record(self, oid: str, sid: str, side: str, result) -> None:
        self.ledger.update_order(oid, result.status, result.broker_order_id, result.message)
        new_qty = result.filled_qty - self.ledger.filled_qty(oid)
        if new_qty > 1e-9 and result.avg_price > 0:
            sign = 1 if side == "BUY" else -1
            cost = self.costs.commission_bps / 10_000 * new_qty * result.avg_price
            self.ledger.add_fill(oid, sid, sign * new_qty, result.avg_price, cost)

    def sync(self, by: str) -> int:
        """Pull status of open orders (asynchronous brokers) and record new fills."""
        updated = 0
        for o in self.ledger.orders():
            if o["status"] in {"FILLED", "REJECTED"} or not o["broker_order_id"]:
                continue
            result = self.broker.order_status(o["broker_order_id"])
            self._record(o["id"], o["security_id"], o["side"], result)
            updated += 1
        self.ledger.audit(by, "ORDERS_SYNCED", {"updated": updated})
        return updated

    # ------------------------------------------------------------------ reconcile
    def reconcile(self, by: str, tolerance: float = 1e-4) -> list[dict]:
        broker_pos = self.broker.positions()
        ledger_pos = self.ledger.implied_positions()
        breaks = []
        for sid in sorted(set(broker_pos) | set(ledger_pos)):
            b, led = broker_pos.get(sid, 0.0), ledger_pos.get(sid, 0.0)
            if abs(b - led) > tolerance:
                kind = "UNTRACKED_AT_BROKER" if sid not in ledger_pos else "QUANTITY_MISMATCH"
                breaks.append({"security_id": sid, "broker": b, "ledger": led, "type": kind})
        account = self.broker.account()
        self.ledger.add_snapshot("broker", account["cash"], account["equity"], broker_pos)
        self.ledger.audit(by, "RECONCILED", {"breaks": breaks})
        return breaks

    # ------------------------------------------------------------------ status
    def status(self) -> dict[str, object]:
        weights, account, positions, prices = self.current_book()
        as_of = self.research.resolve_as_of(None)
        alerts = self.research.alerts(weights, as_of) if len(weights) else []
        history = self.ledger.equity_history()
        if history:
            dd = drawdown_alert(pd.Series([e for _, e in history]))
            if dd:
                alerts.append(dd.to_dict())
        return {
            "controls": self.controls.state(),
            "account": account,
            "positions": {
                s: {"qty": q, "price": prices.get(s), "weight": weights.get(s)}
                for s, q in positions.items()
            },
            "data_as_of": str(as_of.date()),
            "data_age_business_days": self._data_age_bdays(),
            "open_proposals": self.ledger.proposals("PENDING_APPROVAL")
            + self.ledger.proposals("APPROVED"),
            "alerts": alerts,
        }
