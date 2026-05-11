"""Stable canonical-JSON serialization and packet hashing.

Two semantically identical packets must produce identical hashes
regardless of when they were materialized — this is what makes
``packet_hash`` a content fingerprint suitable for replay, dedup, and
A/B comparison.

To achieve that, :func:`compute_packet_hash` excludes the envelope
fields ``packet_id``, ``packet_hash``, and ``generated_at`` before
hashing.  ``packet_id`` is a per-call UUID and ``generated_at`` is a
timestamp; both vary between equivalent calls.

JSON canonicalization rules (`canonical_json`):

* Sort all dict keys (deep).
* Compact separators (no whitespace).
* UUIDs / datetimes / Decimals are serialized via Pydantic's standard
  ``model_dump(mode="json")`` so they end up as strings.
* Floats use Python's standard ``json.dumps`` representation, which is
  deterministic for finite values.

If new envelope fields are added to ``IntelligencePacket`` later, add
them to ``ENVELOPE_FIELDS_EXCLUDED_FROM_HASH`` here.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .packet import IntelligencePacket

ENVELOPE_FIELDS_EXCLUDED_FROM_HASH: frozenset[str] = frozenset(
    {"packet_id", "packet_hash", "generated_at"}
)


def canonical_json(obj: Any) -> str:
    """Return a deterministic JSON string for ``obj``.

    Sorts dict keys recursively, uses compact separators, and
    serializes ``datetime`` / ``UUID`` / ``Decimal`` etc. via Pydantic
    when possible.  For plain dicts/lists the standard library handles
    sorting; for Pydantic models, dump in JSON mode first.
    """
    if hasattr(obj, "model_dump"):
        # Pydantic v2 model. Use mode="json" so UUIDs and datetimes
        # serialize as strings.
        obj = obj.model_dump(mode="json")
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_packet_hash(packet: IntelligencePacket) -> str:
    """Return the sha256 hex digest of the packet's content portion.

    The envelope fields in :data:`ENVELOPE_FIELDS_EXCLUDED_FROM_HASH`
    are excluded so two semantically identical packets hash the same
    regardless of when they were generated or what packet_id was assigned.
    """
    content = packet.model_dump(mode="json", exclude=ENVELOPE_FIELDS_EXCLUDED_FROM_HASH)
    payload = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def attach_packet_hash(packet: IntelligencePacket) -> IntelligencePacket:
    """Compute the packet's content hash and return a copy with
    ``packet_hash`` populated.  The original is left unmodified."""
    new_hash = compute_packet_hash(packet)
    return packet.model_copy(update={"packet_hash": new_hash})
