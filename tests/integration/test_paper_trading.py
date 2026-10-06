"""Sprint 8: paper-trading workflow end to end with the simulated broker."""

from datetime import UTC, datetime, timedelta

import pytest

from apps.api.services.research import ResearchService
from paper_trading.broker import AlpacaPaperBroker, SimulatedBroker
from paper_trading.controls import (
    TradingControls,
    TradingDisabled,
    activate_kill_switch,
    deactivate_kill_switch,
)
from paper_trading.desk import PaperTradingDesk, PreTradeLimits, WorkflowError
from paper_trading.ledger import Ledger


class Clock:
    def __init__(self, start):
        self.now = start

    def __call__(self):
        return self.now


@pytest.fixture
def desk_factory(market, config, tmp_path):
    research = ResearchService(market, config)
    last = market.close.iloc[-1]

    def prices(symbols):
        return {s: float(last[s]) for s in symbols if s in last.index}

    def make(env=None, limits=None, clock=None):
        controls = TradingControls.from_env(
            env if env is not None else {"GLOBAL_TRADING_ENABLED": "true"},
            kill_file=tmp_path / "KILL_SWITCH",
        )
        broker = SimulatedBroker(tmp_path / "sim.json", prices, initial_cash=100_000)
        clock = clock or Clock(datetime(2025, 1, 2, 15, tzinfo=UTC))  # data ends 2024-12-31
        return PaperTradingDesk(
            broker,
            Ledger(tmp_path / "ledger.sqlite"),
            research,
            controls,
            limits or PreTradeLimits(),
            clock,
        )

    return make


def test_full_cycle_propose_approve_execute_reconcile(desk_factory):
    desk = desk_factory()
    pid = desk.propose("analyst1")
    p = desk.ledger.proposal(pid)
    assert p["status"] == "PENDING_APPROVAL"
    assert p["payload"]["pre_trade_ok"]
    assert all(o["qty"] > 0 for o in p["payload"]["orders"])
    assert desk.broker.positions() == {}  # proposing never trades

    with pytest.raises(WorkflowError):
        desk.execute(pid, "pm1")  # not approved
    with pytest.raises(PermissionError):
        desk.approve(pid, "analyst1", role="analyst")

    desk.approve(pid, "boss", role="approver", reason="monthly rebalance")
    counts = desk.execute(pid, "pm1")
    assert counts["filled"] == len(p["payload"]["orders"]) and counts["rejected"] == 0
    assert desk.ledger.proposal(pid)["status"] == "EXECUTED"
    positions = desk.broker.positions()
    assert len(positions) >= 10
    assert desk.broker.account()["cash"] < 5_000  # invested

    with pytest.raises(WorkflowError):
        desk.execute(pid, "pm1")  # never twice
    assert desk.reconcile("ops") == []  # ledger == broker

    actions = [a["action"] for a in desk.ledger.audit_trail(pid)]
    for expected in (
        "PROPOSAL_CREATED",
        "APPROVAL_DENIED",
        "PROPOSAL_APPROVED",
        "PROPOSAL_EXECUTING",
        "PROPOSAL_EXECUTED",
    ):
        assert expected in actions

    status = desk.status()
    assert status["controls"]["global_trading_enabled"] is True
    assert set(status["positions"]) == set(positions)

    # Second cycle starts from the current book: small turnover, no re-buys.
    pid2 = desk.propose("analyst1")
    orders2 = desk.ledger.proposal(pid2)["payload"]["orders"]
    assert len(orders2) < len(p["payload"]["orders"])


def test_global_switch_off_blocks_execution(desk_factory):
    desk = desk_factory(env={})  # GLOBAL_TRADING_ENABLED defaults to false
    pid = desk.propose("a")
    desk.approve(pid, "b", role="approver")
    with pytest.raises(TradingDisabled, match="GLOBAL_TRADING_ENABLED"):
        desk.execute(pid, "c")
    assert desk.broker.positions() == {}
    assert "EXECUTION_BLOCKED" in [a["action"] for a in desk.ledger.audit_trail(pid)]


def test_live_trading_is_refused_even_if_enabled(desk_factory):
    desk = desk_factory(env={"GLOBAL_TRADING_ENABLED": "true", "LIVE_TRADING_ENABLED": "true"})
    pid = desk.propose("a")
    desk.approve(pid, "b", role="approver")
    with pytest.raises(TradingDisabled, match="live trading is not implemented"):
        desk.execute(pid, "c")


def test_kill_switch_blocks_and_can_be_released(desk_factory, tmp_path):
    desk = desk_factory()
    pid = desk.propose("a")
    desk.approve(pid, "b", role="approver")
    activate_kill_switch("market halt drill", "ops", tmp_path / "KILL_SWITCH")
    with pytest.raises(TradingDisabled, match="KILL SWITCH"):
        desk.execute(pid, "c")
    assert deactivate_kill_switch(tmp_path / "KILL_SWITCH")
    desk.execute(pid, "c")
    assert desk.broker.positions()


def test_expired_approval_and_stale_data(desk_factory):
    clock = Clock(datetime(2025, 1, 2, 15, tzinfo=UTC))
    desk = desk_factory(clock=clock)
    pid = desk.propose("a")
    desk.approve(pid, "b", role="approver")
    clock.now += timedelta(hours=30)
    with pytest.raises(WorkflowError, match="approval expired"):
        desk.execute(pid, "c")
    clock.now = datetime(2025, 2, 3, tzinfo=UTC)  # data now weeks old
    with pytest.raises(WorkflowError, match="business days old"):
        desk.propose("a")


def test_pre_trade_limits_block_approval(desk_factory):
    desk = desk_factory(limits=PreTradeLimits(max_order_pct_nav=0.01))
    pid = desk.propose("a")
    payload = desk.ledger.proposal(pid)["payload"]
    assert not payload["pre_trade_ok"]
    with pytest.raises(WorkflowError, match="pre-trade checks failed"):
        desk.approve(pid, "b", role="approver")


def test_four_eyes_rule(desk_factory):
    desk = desk_factory(limits=PreTradeLimits(require_four_eyes=True))
    pid = desk.propose("same_person")
    with pytest.raises(PermissionError, match="four-eyes"):
        desk.approve(pid, "same_person", role="approver")


def test_reconcile_detects_breaks(desk_factory):
    desk = desk_factory()
    pid = desk.propose("a")
    desk.approve(pid, "b", role="approver")
    desk.execute(pid, "c")
    sym = next(iter(desk.broker.positions()))
    desk.broker.state["positions"][sym] += 3  # trade done outside the platform
    desk.broker.state["positions"]["ZZZZ"] = 1
    breaks = {b["security_id"]: b["type"] for b in desk.reconcile("ops")}
    assert breaks == {sym: "QUANTITY_MISMATCH", "ZZZZ": "UNTRACKED_AT_BROKER"}


def test_alpaca_is_pinned_to_paper_and_parses_orders():
    with pytest.raises(ValueError, match="paper endpoint"):
        AlpacaPaperBroker("k", "s", session=object(), base_url="https://api.alpaca.markets")

    class FakeResponse:
        def __init__(self, payload, code=200):
            self.payload, self.status_code, self.text = payload, code, str(payload)

        def json(self):
            return self.payload

        def raise_for_status(self):
            pass

    class FakeSession:
        def __init__(self):
            self.headers, self.posted = {}, []

        def get(self, url, params=None, timeout=None):
            if url.endswith("/v2/account"):
                return FakeResponse({"cash": "1000", "equity": "5000"})
            if url.endswith("/v2/positions"):
                return FakeResponse([{"symbol": "BRK.B", "qty": "2"}])
            if "snapshots" in url:
                return FakeResponse({"BRK.B": {"latestTrade": {"p": 450.5}}})
            return FakeResponse(
                {"id": "o1", "status": "filled", "filled_qty": "2", "filled_avg_price": "10"}
            )

        def post(self, url, json=None, timeout=None):
            self.posted.append(json)
            return FakeResponse({"id": "o1", "status": "accepted"})

    session = FakeSession()
    broker = AlpacaPaperBroker("k", "s", session=session)
    assert session.headers["APCA-API-KEY-ID"] == "k"
    assert broker.account() == {"cash": 1000.0, "equity": 5000.0}
    assert broker.positions() == {"BRK-B": 2.0}
    assert broker.last_prices(["BRK-B"]) == {"BRK-B": 450.5}
    result = broker.submit_market_order("BRK-B", "BUY", 1.5, "ord_1")
    assert result.status == "ACCEPTED"
    assert session.posted[0]["symbol"] == "BRK.B" and session.posted[0]["qty"] == "1.5"
    assert broker.order_status("o1").status == "FILLED"
