from fastapi.testclient import TestClient

from csf_recommendation_engine.main import app


def test_health_endpoints() -> None:
    client = TestClient(app)

    live = client.get("/health/live")
    ready = client.get("/health/ready")

    assert live.status_code == 200
    assert live.json()["status"] == "live"
    assert "x-request-id" in live.headers
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert "x-request-id" in ready.headers
