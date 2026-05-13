"""Local-disk store for demo voice inquiries + LLM-built entity dossiers.

This is a deliberately tiny shim: a JSON file per artifact, loaded fully
on read, rewritten fully on write. No DB, no migrations, no concurrent
write protection. Single-process demo use only.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Serialize writes within a single process so concurrent endpoint calls
# don't shred the JSON file mid-write.
_FILE_LOCK = threading.Lock()


def load_inquiries(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("voice_inquiries file is not a list; ignoring", extra={"path": str(p)})
            return []
        return data
    except json.JSONDecodeError:
        logger.exception("Failed to parse voice_inquiries JSON", extra={"path": str(p)})
        return []


def append_inquiry(
    path: str | Path,
    *,
    entity_id: str,
    ioi_text: str,
    tags: dict[str, Any],
) -> dict[str, Any]:
    """Append a new inquiry record to the JSON file and return it."""
    inquiry = {
        "inquiry_id": f"vi-{uuid.uuid4().hex[:8]}",
        "entity_id": entity_id,
        "ioi_text": ioi_text,
        "tags": tags,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    p = Path(path)
    with _FILE_LOCK:
        existing = load_inquiries(p)
        existing.append(inquiry)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
    return inquiry


def group_inquiries_by_entity(
    inquiries: list[dict[str, Any]],
    entity_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Return entity_id -> [inquiries] for only the given entities."""
    wanted = set(entity_ids)
    out: dict[str, list[dict[str, Any]]] = {eid: [] for eid in entity_ids}
    for inq in inquiries:
        eid = str(inq.get("entity_id", ""))
        if eid in wanted:
            out[eid].append(inq)
    return out


# ---------------------------------------------------------------------------
# Dossier cache
# ---------------------------------------------------------------------------


def load_dossiers(path: str | Path) -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items()}
    except json.JSONDecodeError:
        logger.exception("Failed to parse dossiers JSON", extra={"path": str(p)})
        return {}


def save_dossiers(path: str | Path, dossiers: dict[str, str]) -> None:
    p = Path(path)
    with _FILE_LOCK:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(dossiers, f, indent=2)


def upsert_dossier(path: str | Path, *, entity_id: str, dossier_text: str) -> None:
    with _FILE_LOCK:
        current = load_dossiers(path)
        current[entity_id] = dossier_text
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
