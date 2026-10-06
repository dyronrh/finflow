import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.services.research import ResearchService, get_research_service


@pytest.fixture(scope="module")
def client(market, config):
    service = ResearchService(market, config)
    app.dependency_overrides[get_research_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_rankings(client):
    body = client.get("/v1/rankings", params={"limit": 5}).json()
    assert len(body["items"]) == 5
    scores = [i["composite_score"] for i in body["items"]]
    assert scores == sorted(scores, reverse=True)
    assert body["strategy_version"] == "v0.1.0"
    assert body["data_version"].startswith("synthetic")


def test_rankings_filter_by_decision(client):
    body = client.get("/v1/rankings", params={"decision": "REDUCE", "limit": 500}).json()
    assert body["items"] and all(i["decision"] == "REDUCE" for i in body["items"])


def test_rankings_unknown_date(client):
    assert client.get("/v1/rankings", params={"as_of": "1990-01-01"}).status_code == 404


def test_security_analysis(client):
    sid = client.get("/v1/rankings", params={"limit": 1}).json()["items"][0]["security_id"]
    body = client.get(f"/v1/securities/{sid}/analysis").json()
    assert body["eligible"] is True
    assert body["factor_scores"] is not None
    assert {c["lookback"] for c in body["score_changes"]} == {"1w", "1m", "3m"}
    assert client.get("/v1/securities/NOPE/analysis").status_code == 404


def test_rebalance_proposal_from_cash(client):
    r = client.post("/v1/portfolios/demo/rebalance/proposal", json={"current_nav": 100000})
    body = r.json()
    assert r.status_code == 200
    assert body["status"] == "PENDING_APPROVAL"
    buys = [o for o in body["orders"] if o["side"] == "BUY"]
    assert len(buys) >= 10
    assert sum(o["final_weight"] for o in body["orders"]) == pytest.approx(1.0)
    assert body["estimated_cost_usd"] > 0


def test_rebalance_proposal_rejects_bad_holdings(client):
    r = client.post(
        "/v1/portfolios/demo/rebalance/proposal",
        json={"holdings": {"SEC0001": 0.8, "SEC0002": 0.8}},
    )
    assert r.status_code == 422


def test_proposal_includes_risk_before_after_and_alerts(client):
    body = client.post(
        "/v1/portfolios/demo/rebalance/proposal",
        json={"holdings": {"SEC0001": 0.5, "SEC0002": 0.5}},
    ).json()
    assert body["risk_before"]["volatility_annual"] > 0
    assert "var_95_daily" in body["risk_after"]
    assert isinstance(body["alerts"], list) and isinstance(body["risk_actions"], list)
    # a two-stock book breaches the risk-contribution budget
    assert any(a["code"] == "RISK_CONTRIBUTION" for a in body["alerts"])


def test_portfolio_risk_endpoint(client):
    r = client.post("/v1/portfolios/demo/risk", json={"holdings": {"SEC0001": 0.6, "SEC0002": 0.4}})
    body = r.json()
    assert r.status_code == 200
    assert body["risk"]["gross_exposure"] == pytest.approx(1.0)
    assert sum(body["risk_contributions"].values()) == pytest.approx(1.0, abs=1e-4)
    assert client.post("/v1/portfolios/demo/risk", json={"holdings": {"X": 1}}).status_code == 404
