from __future__ import annotations

import re
from datetime import date
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    # Type-only import to avoid circular dependency between heuristics
    # build (standalone) and the product resolver (domain-level).
    from csf_recommendation_engine.domain.products import ProductResolver


REQUIRED_RAW_COLUMNS = {
    "entity_id",
    "venue",
    "instrument_name",
    "quantity",
    "trade_date",
    "trade_hour",
}


def validate_raw_trade_frame(df_raw: pd.DataFrame) -> None:
    missing = REQUIRED_RAW_COLUMNS.difference(set(df_raw.columns))
    if missing:
        raise ValueError(f"df_raw missing required columns: {sorted(missing)}")


def _require_columns(df: pd.DataFrame, cols: set[str], builder: str) -> None:
    """Per-builder column check. Each new builder needs columns beyond
    the base REQUIRED_RAW_COLUMNS (e.g. ``side``, ``structure``,
    ``status``, ``product_name``). Surface a clean ValueError so the
    nightly job operator can see what's missing without grepping."""
    missing = cols.difference(set(df.columns))
    if missing:
        raise ValueError(f"{builder}: df missing required columns: {sorted(missing)}")


def build_active_venues(
    df_raw: pd.DataFrame, *, lookback_days: int, as_of: date | None = None
) -> pd.DataFrame:
    validate_raw_trade_frame(df_raw)
    if as_of is None:
        as_of = pd.Timestamp.today().date()

    cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=int(lookback_days))
    df_recent = df_raw[df_raw["trade_date"] >= cutoff.date()]

    grouped = (
        df_recent.groupby("entity_id")["venue"]
        .unique()
        .reset_index()
        .rename(columns={"venue": "active_venues"})
    )
    grouped["active_venues"] = grouped["active_venues"].apply(lambda x: [str(v) for v in x])
    return grouped


def build_size_profile(df_raw: pd.DataFrame) -> pd.DataFrame:
    validate_raw_trade_frame(df_raw)
    df_size = (
        df_raw.groupby(["entity_id", "instrument_name"])["quantity"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_size", "std": "stddev_size"})
    )
    df_size["stddev_size"] = df_size["stddev_size"].fillna(0.0)
    return df_size


def build_recency_profile(df_raw: pd.DataFrame) -> pd.DataFrame:
    validate_raw_trade_frame(df_raw)
    df_recency = (
        df_raw.groupby(["entity_id", "instrument_name"])["trade_date"]
        .max()
        .reset_index()
        .rename(columns={"trade_date": "last_trade_date"})
    )
    return df_recency


def build_time_affinity(df_raw: pd.DataFrame) -> pd.DataFrame:
    validate_raw_trade_frame(df_raw)

    df_total_trades = df_raw.groupby("entity_id").size().reset_index(name="total_trades")
    df_hourly = (
        df_raw.groupby(["entity_id", "trade_hour"]).size().reset_index(name="hour_count")
    )

    df_affinity = pd.merge(df_hourly, df_total_trades, on="entity_id")
    df_affinity["hour_ratio"] = df_affinity["hour_count"] / df_affinity["total_trades"]

    df_max_ratio = (
        df_affinity.groupby("entity_id")["hour_ratio"]
        .max()
        .reset_index(name="max_ratio")
    )

    df_affinity = pd.merge(df_affinity, df_max_ratio, on="entity_id")

    def _hourly_ratios(group: pd.DataFrame) -> list[float]:
        ratios = [0.0] * 24
        for _, row in group.iterrows():
            hour = int(row["trade_hour"])
            if 0 <= hour <= 23:
                ratios[hour] = float(row["hour_ratio"])
        return ratios

    df_ratios = (
        df_affinity.groupby("entity_id")
        .apply(_hourly_ratios)
        .reset_index(name="hourly_ratios")
    )

    df_time = pd.merge(df_ratios, df_max_ratio, on="entity_id")
    return df_time


def assemble_entity_features(df_venues: pd.DataFrame, df_time: pd.DataFrame) -> pd.DataFrame:
    df_entity = pd.merge(df_venues, df_time, on="entity_id", how="outer")
    return df_entity


def assemble_instrument_features(df_size: pd.DataFrame, df_recency: pd.DataFrame) -> pd.DataFrame:
    df_inst = pd.merge(df_size, df_recency, on=["entity_id", "instrument_name"], how="outer")
    return df_inst


# ===========================================================================
# Phase 1 / Step 1.1 — Extended L2 features
# ---------------------------------------------------------------------------
# Each builder is a pure pandas/numpy aggregation. None of them touches the
# DB. The nightly job (Step 1.2) is responsible for assembling the raw
# DataFrame, resolving products via ``ProductResolver``, and stitching the
# builder outputs together.
# ===========================================================================


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MONTH_NAME_CODES = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}
_MONTH_LETTER_CODES = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}
_NAME_MONTH_RE = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*(\d{4}|\d{2})\b",
    re.IGNORECASE,
)
_SYMBOL_MONTH_RE = re.compile(r"\b[A-Z]{2,3}([FGHJKMNQUVXZ])(\d{1,2})\b")


def _parse_contract_month(text: str) -> tuple[int, int] | None:
    """Return ``(year, month)`` parsed from an instrument_name / symbol.

    Tries human-readable forms first (``CL Jul26``, ``Heating Oil Feb 26``),
    then CME symbol grammar (``CLN6``, ``HOK6``). Front-leg only.
    Returns ``None`` when no month can be extracted (e.g. ``Cal 26 Strip``).
    """
    if not text:
        return None

    match = _NAME_MONTH_RE.search(text)
    if match:
        m = _MONTH_NAME_CODES[match.group(1)[:3].title()]
        ydigits = match.group(2)
        if len(ydigits) <= 2:
            y = 2020 + int(ydigits) if int(ydigits) < 10 else 2000 + int(ydigits)
        else:
            y = int(ydigits)
        return (y, m)

    match = _SYMBOL_MONTH_RE.search(text)
    if match:
        m = _MONTH_LETTER_CODES[match.group(1)]
        yd = int(match.group(2))
        y = 2020 + yd if yd < 10 else 2000 + yd
        return (y, m)
    return None


def _months_ahead(contract: tuple[int, int], reference: date) -> int:
    """Months from the reference date forward to the contract front leg.

    Negative values mean the contract is already in the past relative to
    the reference date (rare; tolerated and clipped to 0 by callers).
    """
    cy, cm = contract
    return (cy - reference.year) * 12 + (cm - reference.month)


# ---------------------------------------------------------------------------
# build_structure_mix — entity × structure ratios
# ---------------------------------------------------------------------------


def build_structure_mix(df_raw: pd.DataFrame) -> pd.DataFrame:
    """For each entity, return a dict of ``{structure: ratio}`` summing to 1.0.

    ``structure`` values are passed through verbatim from
    ``trade_history.structure``; the current production vocabulary is
    {Flat Price, Spread, Butterfly, Crack, Swap, Unknown}.
    """
    _require_columns(df_raw, {"entity_id", "structure"}, "build_structure_mix")

    if df_raw.empty:
        return pd.DataFrame(columns=["entity_id", "structure_mix"])

    counts = (
        df_raw.groupby(["entity_id", "structure"]).size().reset_index(name="n")
    )
    totals = counts.groupby("entity_id")["n"].sum().reset_index(name="total")
    counts = counts.merge(totals, on="entity_id")
    counts["ratio"] = counts["n"] / counts["total"]

    out_rows: list[dict] = []
    for eid, group in counts.groupby("entity_id"):
        mix = {str(row["structure"]): float(row["ratio"]) for _, row in group.iterrows()}
        out_rows.append({"entity_id": eid, "structure_mix": mix})
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# build_curve_preference — entity × front-month-vs-deferred ratio
# ---------------------------------------------------------------------------


def build_curve_preference(
    df_raw: pd.DataFrame, *, front_month_horizon: int = 2
) -> pd.DataFrame:
    """For each entity, the share of trades whose contract month is
    within ``front_month_horizon`` months of the trade_date — i.e. the
    "front-month preference" vs deferred curve exposure.

    Returns one row per entity with columns
    ``front_month_ratio`` (float in [0,1]) and ``trades_classified``
    (int — total trades that had a parseable contract month).
    Trades whose instrument_name doesn't parse (e.g. ``Cal 26 Strip``)
    are excluded from the denominator.
    """
    _require_columns(
        df_raw, {"entity_id", "instrument_name", "trade_date"}, "build_curve_preference"
    )

    if df_raw.empty:
        return pd.DataFrame(columns=["entity_id", "front_month_ratio", "trades_classified"])

    rows: list[dict] = []
    for _, r in df_raw.iterrows():
        contract = _parse_contract_month(str(r["instrument_name"]))
        if contract is None:
            continue
        ahead = _months_ahead(contract, r["trade_date"])
        is_front = 0 <= ahead <= front_month_horizon
        rows.append({"entity_id": r["entity_id"], "is_front": int(is_front)})

    if not rows:
        return pd.DataFrame(columns=["entity_id", "front_month_ratio", "trades_classified"])

    parsed = pd.DataFrame(rows)
    grouped = (
        parsed.groupby("entity_id")["is_front"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "front_count", "count": "trades_classified"})
    )
    grouped["front_month_ratio"] = grouped["front_count"] / grouped["trades_classified"]
    return grouped[["entity_id", "front_month_ratio", "trades_classified"]]


# ---------------------------------------------------------------------------
# build_flow_polarity — (entity, product) × net BUY-minus-SELL over window
# ---------------------------------------------------------------------------


def build_flow_polarity(
    df_raw: pd.DataFrame,
    *,
    lookback_days: int = 30,
    as_of: date | None = None,
) -> pd.DataFrame:
    """For each (entity, product), net buy-minus-sell volume over the
    last ``lookback_days``. Caller is expected to enrich ``df_raw`` with
    a ``product_name`` column via the ProductResolver before calling.

    Returns a DataFrame with ``entity_id``, ``product_name``, ``net_volume``,
    ``buy_volume``, ``sell_volume``, and ``polarity`` (signed unit-bounded
    value: ``(buy-sell)/(buy+sell)`` in [-1, 1], 0 when both sides are 0).
    """
    _require_columns(
        df_raw,
        {"entity_id", "product_name", "side", "quantity", "trade_date"},
        "build_flow_polarity",
    )

    if df_raw.empty:
        return pd.DataFrame(
            columns=["entity_id", "product_name", "buy_volume", "sell_volume", "net_volume", "polarity"]
        )

    if as_of is None:
        as_of = pd.Timestamp.today().date()
    cutoff = as_of - pd.Timedelta(days=int(lookback_days)).to_pytimedelta()
    df_recent = df_raw[df_raw["trade_date"] >= cutoff]

    df_recent = df_recent.copy()
    df_recent["side_upper"] = df_recent["side"].astype(str).str.upper()

    buy = (
        df_recent[df_recent["side_upper"] == "BUY"]
        .groupby(["entity_id", "product_name"])["quantity"]
        .sum()
        .reset_index(name="buy_volume")
    )
    sell = (
        df_recent[df_recent["side_upper"] == "SELL"]
        .groupby(["entity_id", "product_name"])["quantity"]
        .sum()
        .reset_index(name="sell_volume")
    )

    merged = pd.merge(buy, sell, on=["entity_id", "product_name"], how="outer")
    merged["buy_volume"] = merged["buy_volume"].fillna(0.0).astype(float)
    merged["sell_volume"] = merged["sell_volume"].fillna(0.0).astype(float)
    merged["net_volume"] = merged["buy_volume"] - merged["sell_volume"]
    total = merged["buy_volume"] + merged["sell_volume"]
    merged["polarity"] = np.where(total > 0, merged["net_volume"] / total, 0.0)
    return merged[
        ["entity_id", "product_name", "buy_volume", "sell_volume", "net_volume", "polarity"]
    ]


# ---------------------------------------------------------------------------
# build_follow_through_rate — (entity, product) × filled/(filled+cancelled)
# ---------------------------------------------------------------------------


_FILLED_STATUSES = frozenset({"Filled"})
_AGED_OUT_STATUSES = frozenset({"Cancelled", "Expired", "Aged"})


def build_follow_through_rate(df_raw: pd.DataFrame) -> pd.DataFrame:
    """For each (entity, product), the share of orders that filled vs
    those that aged out (cancelled / expired / aged).

    Returns columns ``entity_id``, ``product_name``,
    ``filled_count``, ``aged_out_count``, ``follow_through_rate``
    (= filled / (filled + aged_out); 0.0 when both are 0).

    Rows whose ``status`` is neither filled nor aged-out (e.g. still
    ``Working``) are excluded from the denominator — the metric is a
    finality-aware quality signal, not a throughput count.
    """
    _require_columns(
        df_raw, {"entity_id", "product_name", "status"}, "build_follow_through_rate"
    )

    if df_raw.empty:
        return pd.DataFrame(
            columns=["entity_id", "product_name", "filled_count", "aged_out_count", "follow_through_rate"]
        )

    df = df_raw.copy()
    df["status_norm"] = df["status"].astype(str).str.title()
    df["is_filled"] = df["status_norm"].isin(_FILLED_STATUSES)
    df["is_aged_out"] = df["status_norm"].isin(_AGED_OUT_STATUSES)

    grouped = (
        df.groupby(["entity_id", "product_name"])[["is_filled", "is_aged_out"]]
        .sum()
        .reset_index()
        .rename(columns={"is_filled": "filled_count", "is_aged_out": "aged_out_count"})
    )
    denom = grouped["filled_count"] + grouped["aged_out_count"]
    grouped["follow_through_rate"] = np.where(
        denom > 0, grouped["filled_count"] / denom, 0.0
    )
    return grouped


# ---------------------------------------------------------------------------
# build_size_profile_at_product — (entity, product) × mean/std qty
# ---------------------------------------------------------------------------


class _ResolverLike(Protocol):
    """Duck-typed interface — anything with ``resolve(name) -> match-or-None``
    whose match has a ``product_name`` attribute satisfies this."""

    def resolve(self, instrument_name: str): ...  # noqa: D401, ANN201


def build_size_profile_at_product(
    df_raw: pd.DataFrame, resolver: "ProductResolver | _ResolverLike"
) -> pd.DataFrame:
    """For each (entity, product), the mean and std of trade quantity.

    The product axis is resolved internally via the supplied
    :class:`ProductResolver` (Step 0.9). Trades whose ``instrument_name``
    cannot be resolved are excluded — the operator-review surface is
    Step 0.9's nightly unresolved-flag job.

    Plan signature exactly: ``build_size_profile_at_product(df, resolver)``.
    """
    _require_columns(
        df_raw,
        {"entity_id", "instrument_name", "quantity"},
        "build_size_profile_at_product",
    )

    if df_raw.empty:
        return pd.DataFrame(
            columns=["entity_id", "product_name", "mean_size_at_product", "stddev_size_at_product"]
        )

    df = df_raw.copy()
    df["product_name"] = df["instrument_name"].apply(
        lambda name: (m.product_name if (m := resolver.resolve(str(name))) else None)
    )
    df = df[df["product_name"].notna()]

    if df.empty:
        return pd.DataFrame(
            columns=["entity_id", "product_name", "mean_size_at_product", "stddev_size_at_product"]
        )

    grouped = (
        df.groupby(["entity_id", "product_name"])["quantity"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_size_at_product", "std": "stddev_size_at_product"})
    )
    grouped["stddev_size_at_product"] = grouped["stddev_size_at_product"].fillna(0.0)
    return grouped


# ---------------------------------------------------------------------------
# build_per_product_liquidity — product × hourly volume bucket
# ---------------------------------------------------------------------------


def build_per_product_liquidity(
    df_raw: pd.DataFrame,
    *,
    lookback_days: int = 30,
    as_of: date | None = None,
) -> pd.DataFrame:
    """Per product, total print volume per hour-of-day across all entities
    in the last ``lookback_days``. Caller pre-resolves ``product_name``.

    Returns columns ``product_name`` and ``hourly_volume`` — a 24-element
    list keyed by hour 0..23.
    """
    _require_columns(
        df_raw,
        {"product_name", "trade_hour", "quantity", "trade_date"},
        "build_per_product_liquidity",
    )

    if df_raw.empty:
        return pd.DataFrame(columns=["product_name", "hourly_volume"])

    if as_of is None:
        as_of = pd.Timestamp.today().date()
    cutoff = as_of - pd.Timedelta(days=int(lookback_days)).to_pytimedelta()
    df = df_raw[df_raw["trade_date"] >= cutoff].copy()

    if df.empty:
        return pd.DataFrame(columns=["product_name", "hourly_volume"])

    df["trade_hour"] = pd.to_numeric(df["trade_hour"], errors="coerce").fillna(-1).astype(int)
    df = df[(df["trade_hour"] >= 0) & (df["trade_hour"] <= 23)]

    grouped = (
        df.groupby(["product_name", "trade_hour"])["quantity"]
        .sum()
        .reset_index(name="hour_volume")
    )

    out_rows: list[dict] = []
    for product, sub in grouped.groupby("product_name"):
        hourly = [0.0] * 24
        for _, row in sub.iterrows():
            hourly[int(row["trade_hour"])] = float(row["hour_volume"])
        out_rows.append({"product_name": product, "hourly_volume": hourly})
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# build_co_activity_matrix — N×N entity pair co-activity
# ---------------------------------------------------------------------------


def build_co_activity_matrix(
    df_raw: pd.DataFrame,
    *,
    entity_universe: list[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Square matrix of pair-wise co-activity in same product on same week.

    ``M[i, j]`` = number of distinct (product, week) buckets in which
    both ``entity_universe[i]`` and ``entity_universe[j]`` placed at
    least one trade. The diagonal is the number of (product, week)
    buckets the entity itself was active in.

    Returns ``(entity_ordering, matrix)``. ``entity_ordering`` lists
    the entity_ids in row/column order. If ``entity_universe`` is None,
    the ordering is the sorted unique entity_ids from ``df_raw``.

    Pure (no DB); caller pre-resolves ``product_name``.
    """
    _require_columns(
        df_raw,
        {"entity_id", "product_name", "trade_date"},
        "build_co_activity_matrix",
    )

    if entity_universe is None:
        ordering = sorted({str(e) for e in df_raw["entity_id"].dropna().unique()})
    else:
        ordering = list(entity_universe)
    n = len(ordering)
    matrix = np.zeros((n, n), dtype=np.int64)

    if df_raw.empty or n == 0:
        return ordering, matrix

    idx_of = {eid: i for i, eid in enumerate(ordering)}

    df = df_raw[["entity_id", "product_name", "trade_date"]].copy()
    df["entity_id"] = df["entity_id"].astype(str)
    df = df[df["entity_id"].isin(idx_of)]
    if df.empty:
        return ordering, matrix

    # Week bucket = ISO year-week pair so cross-year boundaries don't collapse.
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    iso = df["trade_date"].dt.isocalendar()
    df["year_week"] = list(zip(iso["year"], iso["week"]))

    # Distinct (entity, product, week) presence
    presence = df[["entity_id", "product_name", "year_week"]].drop_duplicates()

    # For each (product, week) bucket, take the entity participants and
    # increment all unordered pairs plus the self-bucket count.
    for (_, _), bucket in presence.groupby(["product_name", "year_week"]):
        entities_here = sorted({str(e) for e in bucket["entity_id"]})
        idxs = [idx_of[e] for e in entities_here if e in idx_of]
        for i in idxs:
            matrix[i, i] += 1
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                ia, ib = idxs[a], idxs[b]
                matrix[ia, ib] += 1
                matrix[ib, ia] += 1
    return ordering, matrix
