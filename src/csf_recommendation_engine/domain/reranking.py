from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import exp, log
from typing import Optional

import numpy as np

from csf_recommendation_engine.domain.heuristics_index import HeuristicsIndex


@dataclass(frozen=True)
class RerankWeights:
    lightfm: float
    time_affinity: float
    recency: float
    size_fit: float


def _minmax_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax - vmin <= 1e-12:
        return [0.5 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def _recency_score(*, last_trade_date, now_date, halflife_days: float) -> float:
    if last_trade_date is None:
        return 0.0
    days_ago = max(0, (now_date - last_trade_date).days)
    if halflife_days <= 0:
        return 0.0
    # score = 2^(-days/halflife)
    return float(exp(-log(2) * days_ago / halflife_days))


def _size_fit_score(*, quantity: float, mean_size: Optional[float], stddev_size: Optional[float]) -> float:
    if mean_size is None or quantity is None:
        return 0.0
    if stddev_size is None:
        stddev_size = 0.0

    std = float(stddev_size)
    mean = float(mean_size)

    # Avoid degenerate std=0. Use a conservative floor relative to mean.
    if std <= 0.0:
        std = max(abs(mean) * 0.1, 1.0)

    z = (float(quantity) - mean) / std
    return float(exp(-0.5 * z * z))


def _time_affinity_score(*, ratios: np.ndarray | None, hour: int, max_ratio: float | None) -> float:
    if ratios is None or len(ratios) != 24:
        return 0.0
    if not (0 <= hour <= 23):
        return 0.0
    raw = float(ratios[hour])
    if max_ratio is None or max_ratio <= 0:
        # raw is already a ratio in [0,1]
        return raw
    return min(1.0, raw / float(max_ratio))


def rerank_candidates_with_scores(
    *,
    candidates: list[dict],
    venue: str,
    instrument_name: str,
    quantity: float,
    now_local: datetime,
    heuristics: HeuristicsIndex,
    weights: RerankWeights,
    recency_halflife_days: float,
    apply_venue_filter: bool = True,
) -> list[dict]:
    if not candidates:
        return []

    venue_norm = str(venue)
    instrument_norm = str(instrument_name)

    filtered: list[dict] = []
    for cand in candidates:
        entity_id = str(cand["client_id"])
        if apply_venue_filter:
            venues = heuristics.active_venues_by_entity.get(entity_id)
            if venues is not None and venue_norm not in venues:
                continue
        filtered.append(cand)

    if not filtered:
        return []

    affinities = [float(c["affinity"]) for c in filtered]
    affinity_norms = _minmax_normalize(affinities)

    hour = int(now_local.hour)
    now_date = now_local.date()

    scored: list[dict] = []

    for cand, affinity_norm in zip(filtered, affinity_norms, strict=True):
        entity_id = str(cand["client_id"])

        mean_std = heuristics.size_profile_by_entity_instrument.get((entity_id, instrument_norm))
        mean_size = mean_std[0] if mean_std else None
        stddev_size = mean_std[1] if mean_std else None
        size_score = _size_fit_score(quantity=quantity, mean_size=mean_size, stddev_size=stddev_size)

        last_trade = heuristics.last_trade_date_by_entity_instrument.get((entity_id, instrument_norm))
        recency_score = _recency_score(
            last_trade_date=last_trade,
            now_date=now_date,
            halflife_days=recency_halflife_days,
        )

        ratios = heuristics.hourly_ratios_by_entity.get(entity_id)
        max_ratio = heuristics.max_ratio_by_entity.get(entity_id)
        time_score = _time_affinity_score(ratios=ratios, hour=hour, max_ratio=max_ratio)

        final_score = (
            weights.lightfm * float(affinity_norm)
            + weights.size_fit * float(size_score)
            + weights.recency * float(recency_score)
            + weights.time_affinity * float(time_score)
        )

        scored.append(
            {
                **cand,
                "lightfm_normalized": float(affinity_norm),
                "size_score": float(size_score),
                "recency_score": float(recency_score),
                "time_score": float(time_score),
                "final_score": float(final_score),
            }
        )

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored


def rerank_candidates(
    *,
    candidates: list[dict],
    venue: str,
    instrument_name: str,
    quantity: float,
    now_local: datetime,
    heuristics: HeuristicsIndex,
    weights: RerankWeights,
    recency_halflife_days: float,
    apply_venue_filter: bool = True,
) -> list[dict]:
    return [
        {
            "client_id": cand["client_id"],
            "affinity": cand["affinity"],
        }
        for cand in rerank_candidates_with_scores(
            candidates=candidates,
            venue=venue,
            instrument_name=instrument_name,
            quantity=quantity,
            now_local=now_local,
            heuristics=heuristics,
            weights=weights,
            recency_halflife_days=recency_halflife_days,
            apply_venue_filter=apply_venue_filter,
        )
    ]
