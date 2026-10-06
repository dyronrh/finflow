"""Broker adapters for paper trading (Sprint 8).

* ``SimulatedBroker``: local, deterministic, state persisted to a JSON file.
  Market orders fill immediately at the latest traded price ± slippage.
* ``AlpacaPaperBroker``: Alpaca's paper-trading sandbox over REST. The base
  URL is pinned to the paper endpoint; a live URL is rejected.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from quant_core.execution.costs import TransactionCostModel


@dataclass
class OrderResult:
    broker_order_id: str
    status: str  # FILLED | PARTIALLY_FILLED | ACCEPTED | REJECTED
    filled_qty: float = 0.0
    avg_price: float = 0.0
    message: str = ""


class Broker(Protocol):
    name: str

    def account(self) -> dict[str, float]: ...

    def positions(self) -> dict[str, float]: ...

    def last_prices(self, symbols: list[str]) -> dict[str, float]: ...

    def submit_market_order(
        self, symbol: str, side: str, qty: float, client_order_id: str
    ) -> OrderResult: ...

    def order_status(self, broker_order_id: str) -> OrderResult: ...


# --------------------------------------------------------------------------- simulated
class SimulatedBroker:
    name = "simulated"

    def __init__(
        self,
        state_path: Path,
        price_source: Callable[[list[str]], dict[str, float]],
        initial_cash: float = 100_000.0,
        costs: TransactionCostModel | None = None,
    ) -> None:
        self.state_path = Path(state_path)
        self.price_source = price_source
        self.costs = costs or TransactionCostModel()
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
        else:
            self.state = {"cash": initial_cash, "positions": {}, "orders": {}}
            self._save()

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2))

    def account(self) -> dict[str, float]:
        positions = self.positions()
        prices = self.last_prices(list(positions)) if positions else {}
        equity = self.state["cash"] + sum(q * prices.get(s, 0.0) for s, q in positions.items())
        return {"cash": float(self.state["cash"]), "equity": float(equity)}

    def positions(self) -> dict[str, float]:
        return {s: float(q) for s, q in self.state["positions"].items() if abs(q) > 1e-9}

    def last_prices(self, symbols: list[str]) -> dict[str, float]:
        return self.price_source(symbols)

    def submit_market_order(
        self, symbol: str, side: str, qty: float, client_order_id: str
    ) -> OrderResult:
        oid = f"sim_{uuid.uuid4().hex[:10]}"
        price = self.last_prices([symbol]).get(symbol)
        held = self.state["positions"].get(symbol, 0.0)
        if price is None or price <= 0:
            result = OrderResult(oid, "REJECTED", message="no price")
        elif side == "SELL" and qty > held + 1e-9:
            result = OrderResult(oid, "REJECTED", message=f"insufficient position ({held})")
        else:
            slip = self.costs.slippage_bps / 10_000
            fill = price * (1 + slip) if side == "BUY" else price * (1 - slip)
            notional = qty * fill
            cost = self.costs.commission_bps / 10_000 * notional
            if side == "BUY" and notional + cost > self.state["cash"] + 1e-6:
                result = OrderResult(oid, "REJECTED", message="insufficient cash")
            else:
                sign = 1 if side == "BUY" else -1
                self.state["positions"][symbol] = held + sign * qty
                self.state["cash"] += -sign * notional - cost
                result = OrderResult(oid, "FILLED", qty, fill)
        self.state["orders"][oid] = {**result.__dict__, "client_order_id": client_order_id}
        self._save()
        return result

    def order_status(self, broker_order_id: str) -> OrderResult:
        data = self.state["orders"][broker_order_id]
        return OrderResult(
            broker_order_id, data["status"], data["filled_qty"], data["avg_price"], data["message"]
        )


# --------------------------------------------------------------------------- alpaca
ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"


class AlpacaPaperBroker:
    """Alpaca paper account. Keys: ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY."""

    name = "alpaca_paper"

    def __init__(
        self,
        key_id: str | None = None,
        secret: str | None = None,
        session=None,
        base_url: str = ALPACA_PAPER_URL,
    ):
        if base_url.rstrip("/") != ALPACA_PAPER_URL:
            raise ValueError("only Alpaca's paper endpoint is allowed")
        key_id = key_id or os.environ.get("ALPACA_API_KEY_ID", "")
        secret = secret or os.environ.get("ALPACA_API_SECRET_KEY", "")
        if not key_id or not secret:
            raise ValueError("set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY (paper account keys)")
        if session is None:
            import requests

            session = requests.Session()
        self.session = session
        self.session.headers.update({"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret})
        self.base = base_url.rstrip("/")

    def _get(self, url: str, **params):
        r = self.session.get(url, params=params or None, timeout=30)
        r.raise_for_status()
        return r.json()

    def account(self) -> dict[str, float]:
        a = self._get(f"{self.base}/v2/account")
        return {"cash": float(a["cash"]), "equity": float(a["equity"])}

    def positions(self) -> dict[str, float]:
        return {
            p["symbol"].replace(".", "-"): float(p["qty"])
            for p in self._get(f"{self.base}/v2/positions")
        }

    def last_prices(self, symbols: list[str]) -> dict[str, float]:
        if not symbols:
            return {}
        alp = [s.replace("-", ".") for s in symbols]
        data = self._get(f"{ALPACA_DATA_URL}/v2/stocks/snapshots", symbols=",".join(alp))
        out = {}
        for sym, snap in data.items():
            trade = (snap or {}).get("latestTrade") or {}
            if trade.get("p"):
                out[sym.replace(".", "-")] = float(trade["p"])
        return out

    def submit_market_order(
        self, symbol: str, side: str, qty: float, client_order_id: str
    ) -> OrderResult:
        body = {
            "symbol": symbol.replace("-", "."),
            "qty": f"{qty:.6f}".rstrip("0").rstrip("."),
            "side": side.lower(),
            "type": "market",
            "time_in_force": "day",  # required by Alpaca for fractional quantities
            "client_order_id": client_order_id,
        }
        r = self.session.post(f"{self.base}/v2/orders", json=body, timeout=30)
        if r.status_code >= 400:
            return OrderResult("", "REJECTED", message=r.text[:300])
        return self._parse(r.json())

    def order_status(self, broker_order_id: str) -> OrderResult:
        return self._parse(self._get(f"{self.base}/v2/orders/{broker_order_id}"))

    @staticmethod
    def _parse(o: dict) -> OrderResult:
        status = {
            "filled": "FILLED",
            "partially_filled": "PARTIALLY_FILLED",
            "rejected": "REJECTED",
            "canceled": "REJECTED",
            "expired": "REJECTED",
        }.get(o.get("status", ""), "ACCEPTED")
        return OrderResult(
            o.get("id", ""),
            status,
            float(o.get("filled_qty") or 0.0),
            float(o.get("filled_avg_price") or 0.0),
            o.get("status", ""),
        )
