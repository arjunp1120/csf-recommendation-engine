from fastapi.testclient import TestClient

from csf_recommendation_engine.main import app


def test_shadow_serialize_returns_structured_error_when_uninitialized() -> None:
    client = TestClient(app)

    response = client.post("/internal/shadow/serialize", headers={"x-request-id": "req-789"})

    assert response.status_code == 404
    payload = response.json()["detail"]
    assert payload["code"] == "SHADOW_MODEL_NOT_INITIALIZED"
    assert payload["request_id"] == "req-789"


def test_shadow_serialize_returns_structured_acceptance_when_initialized() -> None:
    client = TestClient(app)

    app.state.shadow_model = object()
    response = client.post("/internal/shadow/serialize", headers={"x-request-id": "req-790"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "accepted"
    assert payload["request_id"] == "req-790"
    assert payload["shadow_model_present"] is True

    app.state.shadow_model = None
