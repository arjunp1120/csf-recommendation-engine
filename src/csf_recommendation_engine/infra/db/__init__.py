"""Database infrastructure.

``pool.py``: connection pool init / getter / dependency.

Per-table modules expose typed async helpers. The intelligence-engine
extended plan (§0.8) adds modules for the new tables and ALTERs:

  - ``ai_engagement_events`` (Step 0.7)
  - ``instrument_products`` (Step 0.8)
  - ``recommendation_serves`` (Step 0.8)
  - ``entity_dossiers`` (Step 0.8; ``nearest_neighbors`` deferred with pgvector)
  - ``market_briefings`` (Step 0.8)
  - ``correlative_attribution`` (Step 0.8)
  - ``voice_inquiries`` (Step 0.8; operational columns from §15.8)
  - ``exposure`` (Step 0.8)
  - ``market_data`` (Step 0.8; ``market_snapshots`` + ``market_color``)
  - ``cross_block_matches`` (Step 0.8; new-shape inserts + dedup queries)

Legacy bulk helpers for ``cross_block_matches`` / ``ai_recommendations``
remain in ``trades.py`` for the nightly batch path.
"""

from csf_recommendation_engine.infra.db import (
    ai_engagement_events,
    client_entities,
    correlative_attribution,
    cross_block_matches,
    entity_dossiers,
    exposure,
    instrument_products,
    market_briefings,
    market_data,
    model_registry,
    pool,
    recommendation_serves,
    trade_history,
    trades,
    voice_inquiries,
)

__all__ = [
    "ai_engagement_events",
    "client_entities",
    "correlative_attribution",
    "cross_block_matches",
    "entity_dossiers",
    "exposure",
    "instrument_products",
    "market_briefings",
    "market_data",
    "model_registry",
    "pool",
    "recommendation_serves",
    "trade_history",
    "trades",
    "voice_inquiries",
]
