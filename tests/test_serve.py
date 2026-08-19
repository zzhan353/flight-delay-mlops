"""API contract tests.

These run against the real trained model in `models/candidate`, so they double as a
smoke test that the artifact loads and scores — the exact failure the deployment
pipeline needs to catch before traffic reaches a new image.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from flight_delay.serve import app

pytestmark = pytest.mark.skipif(
    not Path("models/candidate").exists(),
    reason="no trained model on disk; run `make train` first",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_a_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded_ms"] is not None


def test_predict_returns_a_probability(client):
    resp = client.post(
        "/api/predict",
        json={
            "carrier": "AA",
            "origin": "JFK",
            "dest": "LAX",
            "flight_date": "2025-07-15",
            "dep_hour": 17,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["delay_probability"] <= 1.0
    assert body["route"] == "JFK-LAX"
    assert body["relative_risk"] > 0


def test_evening_departures_are_riskier_than_early_morning(client):
    """A sanity check on the model's behaviour, not just the plumbing.

    Delay propagates through the day, so an identical route leaving at 20:00 must not
    look safer than one leaving at 06:00. If this ever fails, something upstream —
    feature ordering, encoder mapping — has silently broken.
    """

    def prob(hour):
        return client.post(
            "/api/predict",
            json={
                "carrier": "AA",
                "origin": "ORD",
                "dest": "LGA",
                "flight_date": "2025-07-15",
                "dep_hour": hour,
            },
        ).json()["delay_probability"]

    assert prob(20) > prob(6)


def test_unknown_route_is_rejected_with_an_explanation(client):
    resp = client.post(
        "/api/predict",
        json={
            "carrier": "AA",
            "origin": "JFK",
            "dest": "ZZZ",
            "flight_date": "2025-07-15",
            "dep_hour": 9,
        },
    )
    assert resp.status_code == 404
    assert "never seen" in resp.json()["detail"]


def test_invalid_hour_is_rejected(client):
    resp = client.post(
        "/api/predict",
        json={
            "carrier": "AA",
            "origin": "JFK",
            "dest": "LAX",
            "flight_date": "2025-07-15",
            "dep_hour": 99,
        },
    )
    assert resp.status_code == 422


def test_routes_endpoint_only_lists_served_destinations(client):
    body = client.get("/api/routes/ORD").json()
    assert "LGA" in body["destinations"]
    assert "ZZZ" not in body["destinations"]


def test_index_serves_the_demo_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Will this flight arrive late?" in resp.text
