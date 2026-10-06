"""SQLite trade ledger with an append-only audit log (README §2.3, §15.4).

Every sensitive action (proposal, approval, order, fill, reconciliation,
kill switch) is written to ``audit_log`` with who, when and what. Rows in
``audit_log`` and ``fills`` are never updated or deleted.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS proposals (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    created_by TEXT NOT NULL,
    as_of TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    data_version TEXT NOT NULL,
    broker TEXT NOT NULL,
    nav REAL NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
    proposal_id TEXT NOT NULL REFERENCES proposals(id),
    decided_at TEXT NOT NULL,
    decided_by TEXT NOT NULL,
    role TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL REFERENCES proposals(id),
    security_id TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    est_price REAL NOT NULL,
    status TEXT NOT NULL,
    broker_order_id TEXT,
    submitted_at TEXT,
    message TEXT
);
CREATE TABLE IF NOT EXISTS fills (
    order_id TEXT NOT NULL REFERENCES orders(id),
    filled_at TEXT NOT NULL,
    security_id TEXT NOT NULL,
    qty REAL NOT NULL,
    price REAL NOT NULL,
    cost REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    taken_at TEXT NOT NULL,
    source TEXT NOT NULL,
    cash REAL,
    equity REAL,
    positions TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    at TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    proposal_id TEXT,
    details TEXT NOT NULL
);
"""


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Ledger:
    def __init__(self, path: Path, clock=_utcnow) -> None:
        self.clock = clock
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def now(self) -> str:
        return self.clock().isoformat()

    # --- audit -----------------------------------------------------------------
    def audit(self, actor: str, action: str, details: dict, proposal_id: str | None = None) -> None:
        self.db.execute(
            "INSERT INTO audit_log VALUES (?, ?, ?, ?, ?)",
            (self.now(), actor, action, proposal_id, json.dumps(details, default=str)),
        )
        self.db.commit()

    def audit_trail(self, proposal_id: str | None = None) -> list[dict]:
        sql = "SELECT * FROM audit_log"
        args: tuple = ()
        if proposal_id:
            sql += " WHERE proposal_id = ?"
            args = (proposal_id,)
        return [dict(r) for r in self.db.execute(sql + " ORDER BY rowid", args)]

    # --- proposals -------------------------------------------------------------
    def add_proposal(self, payload: dict, created_by: str, broker: str) -> str:
        pid = f"rb_{payload['as_of_date']}_{uuid.uuid4().hex[:8]}"
        self.db.execute(
            "INSERT INTO proposals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                pid,
                self.now(),
                created_by,
                str(payload["as_of_date"]),
                payload["strategy_version"],
                payload["data_version"],
                broker,
                float(payload["current_nav"]),
                "PENDING_APPROVAL",
                json.dumps(payload, default=str),
            ),
        )
        self.db.commit()
        self.audit(created_by, "PROPOSAL_CREATED", {"orders": len(payload["orders"])}, pid)
        return pid

    def proposal(self, pid: str) -> dict:
        row = self.db.execute("SELECT * FROM proposals WHERE id = ?", (pid,)).fetchone()
        if row is None:
            raise KeyError(f"unknown proposal {pid}")
        out = dict(row)
        out["payload"] = json.loads(out["payload"])
        return out

    def proposals(self, status: str | None = None) -> list[dict]:
        sql, args = "SELECT id, created_at, as_of, status, nav, strategy_version FROM proposals", ()
        if status:
            sql, args = sql + " WHERE status = ?", (status,)
        return [dict(r) for r in self.db.execute(sql + " ORDER BY created_at", args)]

    def set_proposal_status(self, pid: str, status: str, actor: str, details: dict | None = None):
        self.db.execute("UPDATE proposals SET status = ? WHERE id = ?", (status, pid))
        self.db.commit()
        self.audit(actor, f"PROPOSAL_{status}", details or {}, pid)

    def add_approval(self, pid: str, by: str, role: str, decision: str, reason: str | None):
        self.db.execute(
            "INSERT INTO approvals VALUES (?, ?, ?, ?, ?, ?)",
            (pid, self.now(), by, role, decision, reason),
        )
        self.db.commit()

    def approval(self, pid: str) -> dict | None:
        row = self.db.execute(
            "SELECT * FROM approvals WHERE proposal_id = ? ORDER BY rowid DESC LIMIT 1", (pid,)
        ).fetchone()
        return dict(row) if row else None

    # --- orders and fills ------------------------------------------------------
    def add_order(self, pid: str, security_id: str, side: str, qty: float, est_price: float) -> str:
        oid = f"ord_{uuid.uuid4().hex[:12]}"
        self.db.execute(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)",
            (oid, pid, security_id, side, qty, est_price, "NEW"),
        )
        self.db.commit()
        return oid

    def update_order(
        self, oid: str, status: str, broker_order_id: str | None = None, message: str | None = None
    ):
        self.db.execute(
            "UPDATE orders SET status = ?, broker_order_id = COALESCE(?, broker_order_id), "
            "submitted_at = COALESCE(submitted_at, ?), message = ? WHERE id = ?",
            (status, broker_order_id, self.now(), message, oid),
        )
        self.db.commit()

    def orders(self, pid: str | None = None, status: str | None = None) -> list[dict]:
        sql, clauses, args = "SELECT * FROM orders", [], []
        if pid:
            clauses.append("proposal_id = ?")
            args.append(pid)
        if status:
            clauses.append("status = ?")
            args.append(status)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return [dict(r) for r in self.db.execute(sql, args)]

    def add_fill(self, oid: str, security_id: str, qty: float, price: float, cost: float) -> None:
        self.db.execute(
            "INSERT INTO fills VALUES (?, ?, ?, ?, ?, ?)",
            (oid, self.now(), security_id, qty, price, cost),
        )
        self.db.commit()

    def filled_qty(self, oid: str) -> float:
        row = self.db.execute(
            "SELECT COALESCE(SUM(ABS(qty)), 0) FROM fills WHERE order_id = ?", (oid,)
        )
        return float(row.fetchone()[0])

    def implied_positions(self) -> dict[str, float]:
        """Positions implied by every fill in the ledger (signed quantities)."""
        rows = self.db.execute("SELECT security_id, SUM(qty) FROM fills GROUP BY security_id")
        return {sid: float(q) for sid, q in rows if abs(q) > 1e-9}

    # --- snapshots -------------------------------------------------------------
    def add_snapshot(self, source: str, cash: float, equity: float, positions: dict) -> None:
        self.db.execute(
            "INSERT INTO snapshots VALUES (?, ?, ?, ?, ?)",
            (self.now(), source, cash, equity, json.dumps(positions)),
        )
        self.db.commit()

    def equity_history(self) -> list[tuple[str, float]]:
        rows = self.db.execute("SELECT taken_at, equity FROM snapshots WHERE source = 'broker'")
        return [(r[0], float(r[1])) for r in rows if r[1] is not None]
