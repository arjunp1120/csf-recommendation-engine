from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import asyncpg
import pandas as pd


@dataclass(frozen=True)
class TradeHistoryColumnConfig:
    entity_id: str
    venue: str
    instrument_name: str
    quantity: str
    status: str
    trade_date: str | None
    trade_time: str | None
    trade_ts: str | None


async def _fetch_column_metadata(
    conn: asyncpg.Connection, *, schema: str, table: str
) -> dict[str, dict[str, str]]:
    rows = await conn.fetch(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = $1 AND table_name = $2
        """,
        schema,
        table,
    )
    return {str(r["column_name"]): {"data_type": str(r["data_type"])} for r in rows}


def _pick_first(available: set[str], candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if name in available:
            return name
    return None


async def detect_trade_history_columns(
    conn: asyncpg.Connection, *, schema: str = "public", table: str = "trade_history"
) -> TradeHistoryColumnConfig:
    meta = await _fetch_column_metadata(conn, schema=schema, table=table)
    available = set(meta.keys())

    entity_id = _pick_first(available, ["entity_id", "client_id", "counterparty_id"])
    venue = _pick_first(available, ["venue", "execution_venue"])
    instrument_name = _pick_first(available, ["instrument_name", "instrument", "product"])
    quantity = _pick_first(available, ["quantity", "qty", "size"])
    status = _pick_first(available, ["status", "trade_status"])

    if not entity_id or not venue or not instrument_name or not quantity or not status:
        missing = [
            ("entity_id", entity_id),
            ("venue", venue),
            ("instrument_name", instrument_name),
            ("quantity", quantity),
            ("status", status),
        ]
        missing_names = [label for label, value in missing if value is None]
        raise RuntimeError(
            f"trade_history column detection failed; missing: {', '.join(missing_names)}"
        )

    trade_ts = _pick_first(
        available,
        [
            "trade_ts",
            "trade_timestamp",
            "trade_datetime",
            "trade_time",
            "execution_timestamp",
        ],
    )

    trade_date = _pick_first(available, ["trade_date", "execution_date", "date"])

    trade_time: str | None = None
    if trade_ts == "trade_time":
        # If trade_time exists, it could be a timestamp or a time-only column.
        # If trade_date is also present, we assume trade_time is time-only.
        if trade_date is not None and meta.get("trade_time", {}).get("data_type") == "time without time zone":
            trade_time = "trade_time"
            trade_ts = None

    if trade_ts is None:
        trade_time = trade_time or _pick_first(available, ["trade_time", "execution_time", "time"])

    if trade_ts is None and (trade_date is None or trade_time is None):
        raise RuntimeError(
            "trade_history must have either a timestamp column (trade_ts/trade_timestamp/...) "
            "or both trade_date and trade_time"
        )

    return TradeHistoryColumnConfig(
        entity_id=entity_id,
        venue=venue,
        instrument_name=instrument_name,
        quantity=quantity,
        status=status,
        trade_date=trade_date,
        trade_time=trade_time,
        trade_ts=trade_ts,
    )


async def fetch_trade_history_frame(
    conn: asyncpg.Connection,
    *,
    completed_statuses: list[str],
    history_days: int | None = 365,
    schema: str = "public",
    table: str = "trade_history",
    limit: int | None = None,
) -> pd.DataFrame:
    cols = await detect_trade_history_columns(conn, schema=schema, table=table)

    where_clauses: list[str] = [f"{cols.status} = ANY($1::text[])"]
    params: list[object] = [completed_statuses]

    if history_days is not None:
        if cols.trade_ts is not None:
            where_clauses.append(f"{cols.trade_ts}::date >= (CURRENT_DATE - $2::int)")
        else:
            where_clauses.append(f"{cols.trade_date}::date >= (CURRENT_DATE - $2::int)")
        params.append(int(history_days))

    where_sql = " AND ".join(where_clauses)

    if cols.trade_ts is not None:
        select_sql = f"""
            SELECT
                {cols.entity_id}::text AS entity_id,
                {cols.venue}::text AS venue,
                {cols.instrument_name}::text AS instrument_name,
                {cols.quantity}::double precision AS quantity,
                ({cols.trade_ts}::date) AS trade_date,
                EXTRACT(HOUR FROM {cols.trade_ts})::int AS trade_hour
            FROM {schema}.{table}
            WHERE {where_sql}
        """
    else:
        select_sql = f"""
            SELECT
                {cols.entity_id}::text AS entity_id,
                {cols.venue}::text AS venue,
                {cols.instrument_name}::text AS instrument_name,
                {cols.quantity}::double precision AS quantity,
                ({cols.trade_date}::date) AS trade_date,
                EXTRACT(HOUR FROM {cols.trade_time})::int AS trade_hour
            FROM {schema}.{table}
            WHERE {where_sql}
        """

    if limit is not None and limit > 0:
        select_sql += f"\nLIMIT {int(limit)}"

    rows = await conn.fetch(select_sql, *params)
    return pd.DataFrame([dict(r) for r in rows])
