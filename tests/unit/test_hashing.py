"""Unit tests for canonical-JSON + packet hashing (plan Step 0.4, §12).

Verifies the four properties hashing exists to guarantee:

  1. Same content + different envelope (packet_id / generated_at /
     packet_hash) → SAME hash.
  2. Different content → DIFFERENT hash.
  3. ``canonical_json`` is order-independent for dict inputs.
  4. ``attach_packet_hash`` returns a *new* packet (original unmodified).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from csf_recommendation_engine.domain.intelligence.hashing import (
    ENVELOPE_FIELDS_EXCLUDED_FROM_HASH,
    attach_packet_hash,
    canonical_json,
    compute_packet_hash,
)
from csf_recommendation_engine.domain.intelligence.packet import (
    IntelligencePacket,
    RequestContext,
)


# ---------------------------------------------------------------------------
# canonical_json
# ---------------------------------------------------------------------------


class TestCanonicalJson:
    def test_dict_key_order_independence(self) -> None:
        a = {"b": 1, "a": 2, "c": {"y": 3, "x": 4}}
        b = {"c": {"x": 4, "y": 3}, "a": 2, "b": 1}
        assert canonical_json(a) == canonical_json(b)

    def test_compact_separators_no_whitespace(self) -> None:
        out = canonical_json({"a": 1, "b": [2, 3]})
        assert " " not in out

    def test_handles_pydantic_model(self) -> None:
        rc = RequestContext(request_type="sync_form", current_hour=14)
        out = canonical_json(rc)
        # round-trip parses cleanly
        parsed = json.loads(out)
        assert parsed["request_type"] == "sync_form"
        assert parsed["current_hour"] == 14


# ---------------------------------------------------------------------------
# Envelope inclusion check
# ---------------------------------------------------------------------------


class TestEnvelopeExclusionSet:
    def test_canonical_set(self) -> None:
        assert ENVELOPE_FIELDS_EXCLUDED_FROM_HASH == frozenset(
            {"packet_id", "packet_hash", "generated_at"}
        )


# ---------------------------------------------------------------------------
# compute_packet_hash
# ---------------------------------------------------------------------------


def _packet(**overrides: object) -> IntelligencePacket:
    base: dict[str, object] = {
        "packet_id": uuid4(),
        "generated_at": datetime(2026, 5, 13, 14, 0, 0, tzinfo=timezone.utc),
        "request_context": RequestContext(request_type="sync_form", current_hour=14),
    }
    base.update(overrides)
    return IntelligencePacket(**base)  # type: ignore[arg-type]


class TestComputePacketHash:
    def test_deterministic_for_same_content(self) -> None:
        p1 = _packet(packet_id=uuid4(), generated_at=datetime.now(timezone.utc))
        p2 = _packet(packet_id=uuid4(), generated_at=datetime.now(timezone.utc) + timedelta(hours=5))
        assert compute_packet_hash(p1) == compute_packet_hash(p2)

    def test_packet_id_excluded(self) -> None:
        h_a = compute_packet_hash(_packet(packet_id=uuid4()))
        h_b = compute_packet_hash(_packet(packet_id=uuid4()))
        assert h_a == h_b

    def test_generated_at_excluded(self) -> None:
        h_a = compute_packet_hash(_packet(generated_at=datetime(2020, 1, 1, tzinfo=timezone.utc)))
        h_b = compute_packet_hash(_packet(generated_at=datetime(2050, 12, 31, tzinfo=timezone.utc)))
        assert h_a == h_b

    def test_packet_hash_field_excluded(self) -> None:
        p1 = _packet()
        p1_with_hash = p1.model_copy(update={"packet_hash": "deadbeef" * 8})
        assert compute_packet_hash(p1) == compute_packet_hash(p1_with_hash)

    def test_content_change_changes_hash(self) -> None:
        p1 = _packet(request_context=RequestContext(request_type="sync_form", current_hour=14))
        p2 = _packet(request_context=RequestContext(request_type="sync_form", current_hour=15))
        assert compute_packet_hash(p1) != compute_packet_hash(p2)

    def test_request_type_change_changes_hash(self) -> None:
        p1 = _packet(request_context=RequestContext(request_type="sync_form", current_hour=14))
        p2 = _packet(request_context=RequestContext(request_type="ioi_accept", current_hour=14))
        assert compute_packet_hash(p1) != compute_packet_hash(p2)

    def test_hash_is_sha256_hex(self) -> None:
        h = compute_packet_hash(_packet())
        assert len(h) == 64
        # Round-trip: hex digits only
        int(h, 16)


# ---------------------------------------------------------------------------
# attach_packet_hash
# ---------------------------------------------------------------------------


class TestAttachPacketHash:
    def test_returns_new_packet_with_hash_set(self) -> None:
        p = _packet()
        assert p.packet_hash == ""
        p_with_hash = attach_packet_hash(p)
        assert p_with_hash.packet_hash == compute_packet_hash(p)
        assert len(p_with_hash.packet_hash) == 64

    def test_original_packet_unmodified(self) -> None:
        p = _packet()
        attach_packet_hash(p)
        assert p.packet_hash == ""

    def test_attaching_twice_is_stable(self) -> None:
        """Attaching the hash, then attaching again, gives the same hash —
        because ``packet_hash`` is in the envelope-exclusion set."""
        p1 = attach_packet_hash(_packet())
        p2 = attach_packet_hash(p1)
        assert p1.packet_hash == p2.packet_hash
