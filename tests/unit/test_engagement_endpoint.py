"""Unit tests for ``POST /events/engagement`` (plan Step 0.7).

The DB layer is mocked via FastAPI's ``dependency_overrides`` so the
test never hits Postgres. Covers:

  * 201 happy path with ``serve_id``
  * 201 happy path with ``match_id``
  * 422 zero targets (model-level validator)
  * 422 both targets (model-level validator)
  * 422 invalid ``event_type`` (Literal mismatch)
  * 422 extra fields rejected (``extra='forbid'``)
  * 404 mapping for ``asyncpg.ForeignKeyViolationError``
  * 422 mapping for ``asyncpg.CheckViolationError``
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import asyncpg
import pytest
from fastapi.testclient import TestClient

from csf_recommendation_engine.api.routes.events import router as events_router
from csf_recommendation_engine.infra.db.pool import get_db_connection


# ---------------------------------------------------------------------------
# Test app + dependency override
# ---------------------------------------------------------------------------


def _make_app(mock_conn: Any):
    """Build a minimal FastAPI app that mounts only the events router and
    overrides the DB-connection dependency to yield ``mock_conn``."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(events_router)

    async def _yield_mock_conn():
        yield mock_conn

    app.dependency_overrides[get_db_connection] = _yield_mock_conn
    return app


def _mock_conn(*, returned_event_id: UUID | None = None, raises: Exception | None = None):
    """Build a duck-typed asyncpg-style connection whose ``fetchrow``
    either returns a row-dict or raises an exception.

    Note: do NOT use ``spec=["fetchrow"]`` here — that would make the
    child attribute a plain ``Mock`` (not awaitable) and break the
    ``await conn.fetchrow(...)`` call inside ``insert_event``. Without a
    spec, ``AsyncMock`` auto-creates each child as an ``AsyncMock`` too.
    """
    conn = AsyncMock()
    if raises is not None:
        conn.fetchrow = AsyncMock(side_effect=raises)
    else:
        eid = returned_event_id or uuid4()
        conn.fetchrow = AsyncMock(return_value={"event_id": eid})
    return conn


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_201_with_serve_id(self) -> None:
        event_id = uuid4()
        serve_id = uuid4()
        conn = _mock_conn(returned_event_id=event_id)
        app = _make_app(conn)

        with TestClient(app) as client:
            r = client.post("/events/engagement", json={
                "event_type": "thumbs_up",
                "serve_id": str(serve_id),
                "metadata": {"note": "smoke"},
            })
        assert r.status_code == 201, r.text
        assert r.json() == {"event_id": str(event_id)}
        conn.fetchrow.assert_awaited_once()

    def test_201_with_match_id(self) -> None:
        event_id = uuid4()
        match_id = uuid4()
        conn = _mock_conn(returned_event_id=event_id)
        app = _make_app(conn)

        with TestClient(app) as client:
            r = client.post("/events/engagement", json={
                "event_type": "impression",
                "match_id": str(match_id),
            })
        assert r.status_code == 201
        assert r.json() == {"event_id": str(event_id)}


# ---------------------------------------------------------------------------
# 422 — payload validation errors
# ---------------------------------------------------------------------------


class TestPayloadValidation:
    def test_zero_targets_422(self) -> None:
        conn = _mock_conn()
        app = _make_app(conn)
        with TestClient(app) as client:
            r = client.post("/events/engagement", json={
                "event_type": "click",
                # neither serve_id nor match_id
            })
        assert r.status_code == 422
        conn.fetchrow.assert_not_awaited()

    def test_both_targets_422(self) -> None:
        conn = _mock_conn()
        app = _make_app(conn)
        with TestClient(app) as client:
            r = client.post("/events/engagement", json={
                "event_type": "click",
                "serve_id": str(uuid4()),
                "match_id": str(uuid4()),
            })
        assert r.status_code == 422
        conn.fetchrow.assert_not_awaited()

    def test_invalid_event_type_422(self) -> None:
        conn = _mock_conn()
        app = _make_app(conn)
        with TestClient(app) as client:
            r = client.post("/events/engagement", json={
                "event_type": "not_a_real_event",
                "serve_id": str(uuid4()),
            })
        assert r.status_code == 422

    def test_extra_field_422(self) -> None:
        conn = _mock_conn()
        app = _make_app(conn)
        with TestClient(app) as client:
            r = client.post("/events/engagement", json={
                "event_type": "click",
                "serve_id": str(uuid4()),
                "rogue_extra_field": "boom",
            })
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# DB error mappings
# ---------------------------------------------------------------------------


class TestDBErrorMappings:
    def test_foreign_key_violation_maps_to_404(self) -> None:
        # asyncpg.ForeignKeyViolationError requires a message arg
        fkv = asyncpg.ForeignKeyViolationError("FK target missing")
        conn = _mock_conn(raises=fkv)
        app = _make_app(conn)
        with TestClient(app) as client:
            r = client.post("/events/engagement", json={
                "event_type": "click",
                "serve_id": str(uuid4()),
            })
        assert r.status_code == 404
        assert "not found" in r.json()["detail"].lower()

    def test_check_violation_maps_to_422(self) -> None:
        cv = asyncpg.CheckViolationError("one_target CHECK violated")
        conn = _mock_conn(raises=cv)
        app = _make_app(conn)
        with TestClient(app) as client:
            r = client.post("/events/engagement", json={
                "event_type": "click",
                "serve_id": str(uuid4()),
            })
        assert r.status_code == 422
