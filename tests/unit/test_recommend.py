from __future__ import annotations

from collections.abc import AsyncGenerator

import numpy as np
import pytest
from fastapi.testclient import TestClient

import csf_recommendation_engine.api.routes.recommend as recommend_route
from csf_recommendation_engine.domain.heuristics_index import HeuristicsIndex
from csf_recommendation_engine.infra.db.pool import get_optional_db_connection
from csf_recommendation_engine.main import create_app


class _FakeModel:
    def __init__(self, scores: np.ndarray):
        self._scores = scores

    def predict(self, *, user_ids, item_ids, user_features=None, item_features=None):
        # Return a score per user id in the same order.
        return self._scores[np.asarray(user_ids, dtype=int)]


async def _fake_get_model_data(heuristics: HeuristicsIndex | None) -> dict:
    # 3 users in entity_index: self + two candidates
    entity_index = {"self": 0, "c1": 1, "c2": 2}
    rev_entity = {v: k for k, v in entity_index.items()}

    live = {
        "proxy_index": {"D1_STRUCT-1_SELL": 0},
        "entity_index": entity_index,
        "model": _FakeModel(scores=np.array([0.0, 0.9, 0.8], dtype=float)),
    }
    mats = {"user_features_matrix": None, "item_features_matrix": None}
    return {"live": live, "mats": mats, "rev_entity": rev_entity, "heuristics": heuristics}


async def _fake_db_conn_override() -> AsyncGenerator[None, None]:
    # Yielding None triggers the "Unknown Entity" fallback.
    yield None


def _base_payload(**overrides):
    payload = {
        "client_id": "self",
        "desk": "D1",
        "structure": "STRUCT 1",
        "side": "BUY",
        "venue": "ICE",
        "instrument_name": "CL",
        "quantity": 100,
        "top_k": 2,
    }
    payload.update(overrides)
    return payload


def test_recommend_requires_new_fields() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.post(
        "/recommend",
        json={
            "client_id": "self",
            "desk": "D1",
            "structure": "STRUCT 1",
            "side": "BUY",
        },
    )
    assert response.status_code == 422


def test_recommend_returns_503_when_model_unloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    app = create_app()
    app.dependency_overrides[get_optional_db_connection] = _fake_db_conn_override
    client = TestClient(app)

    async def _unloaded():
        raise ValueError("Model data has not been preloaded")

    monkeypatch.setattr(recommend_route, "get_model_data", _unloaded)

    response = client.post("/recommend", json=_base_payload())
    assert response.status_code == 503


def test_recommend_applies_active_venue_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    # c1 is NOT active on ICE; c2 IS active on ICE.
    heuristics = HeuristicsIndex(
        active_venues_by_entity={"c1": {"CME"}, "c2": {"ICE"}},
        hourly_ratios_by_entity={},
        max_ratio_by_entity={},
        size_profile_by_entity_instrument={},
        last_trade_date_by_entity_instrument={},
    )

    async def _loaded():
        return await _fake_get_model_data(heuristics)

    monkeypatch.setattr(recommend_route, "get_model_data", _loaded)

    app = create_app()
    app.dependency_overrides[get_optional_db_connection] = _fake_db_conn_override
    client = TestClient(app)

    response = client.post("/recommend", json=_base_payload(venue="ICE"))
    assert response.status_code == 200

    payload = response.json()
    ids = [c["client_id"] for c in payload["counterparties"]]
    assert "c1" not in ids
    assert "c2" in ids
