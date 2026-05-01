from fastapi.testclient import TestClient

from csf_recommendation_engine.main import app


def test_admin_approval_endpoint_remains_scaffolded() -> None:
    client = TestClient(app)

    response = client.post("/admin/models/model-123/approve")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "pending-implementation"
    assert payload["model_id"] == "model-123"
    assert "deferred" in payload["detail"].lower()
