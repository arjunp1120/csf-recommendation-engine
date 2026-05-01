"""Standalone ETL script for building proxy-model training artifacts from trade_history."""

from __future__ import annotations

import asyncio
import pickle
import re
import ssl
from pathlib import Path

import asyncpg
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from csf_recommendation_engine.core.config import get_settings

settings = get_settings()
POSTGRES_DSN = settings.postgres_dsn

DIR_PATH = Path(__file__).parent
# EXECUTED_STATUS_EXCLUSIONS = {
#     "Rejected",
#     "Suspended",
#     "Pending New",
# }
INCLUDED_STATUSSES = {
    "Filled", "Calculated", "Done for Day", "Pending Clearing"
}
INCLUDED_SECONDARY_STATUSSES = {
    "Expired", "Working", "Stopped"
}
INCLUDED_TERTIARY_STATUSSES = {
    "Cancelled", "Pending Cancel"
}
VOLUME_LABELS = ["vol_Q1", "vol_Q2", "vol_Q3", "vol_Q4"]

VELOCITY_LABELS = ["vel_Low", "vel_Med", "vel_High"]
TRADE_DECAY_LAMBDA = 0.01


def build_ssl_context() -> ssl.SSLContext:
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    return ssl_ctx


def normalize_component(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "-", text)
    return text.strip("-")


def safe_quantile_labels(values: pd.Series, labels: list[str]) -> pd.Series:
    if values.empty:
        return pd.Series(dtype="object")

    clean_values = values.dropna()
    if clean_values.empty:
        return pd.Series([labels[0]] * len(values), index=values.index)

    bin_count = min(len(labels), clean_values.nunique())
    if bin_count <= 1:
        return pd.Series([labels[0]] * len(values), index=values.index)

    try:
        codes = pd.qcut(values, q=bin_count, labels=False, duplicates="drop")
    except ValueError:
        codes = pd.qcut(values.rank(method="first"), q=bin_count, labels=False, duplicates="drop")

    code_series = pd.Series(codes, index=values.index).fillna(0).astype(int)
    return code_series.map(lambda code: labels[min(int(code), len(labels) - 1)])


def write_pickle(path: Path, payload: object) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


async def load_trade_history() -> pd.DataFrame:
    query = """
        SELECT entity_id, trade_date, quantity, price, desk_type, structure, side, status
        FROM trade_history
    """

    connection = await asyncpg.connect(dsn=POSTGRES_DSN, ssl=build_ssl_context())
    try:
        records = await connection.fetch(query)
    finally:
        await connection.close()

    if not records:
        return pd.DataFrame(
            columns=["entity_id", "trade_date", "quantity", "price", "desk_type", "structure", "side", "status"]
        )

    return pd.DataFrame([dict(record) for record in records])


def sanitize_trade_history(raw_trade_history: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    if raw_trade_history.empty:
        raise RuntimeError("trade_history query returned no rows.")

    working = raw_trade_history.copy()
    working["entity_id"] = working["entity_id"].astype(str).str.strip()
    working["trade_date"] = pd.to_datetime(working["trade_date"], errors="coerce")
    working["quantity"] = pd.to_numeric(working["quantity"], errors="coerce")
    working["price"] = pd.to_numeric(working["price"], errors="coerce")
    working["desk_type"] = working["desk_type"].map(normalize_component)
    working["structure"] = working["structure"].map(normalize_component)
    working["side"] = working["side"].map(normalize_component)
    working["status"] = working["status"].astype("string").str.strip()

    primary_statuses = {s.casefold() for s in INCLUDED_STATUSSES}
    secondary_statuses = {s.casefold() for s in INCLUDED_SECONDARY_STATUSSES}
    tertiary_statuses = {s.casefold() for s in INCLUDED_TERTIARY_STATUSSES}
   
    valid_statuses = valid_statuses = primary_statuses | secondary_statuses | tertiary_statuses
    status_col = working["status"].fillna("").str.strip().str.casefold()
    keep_mask = status_col.isin(valid_statuses)
    working = working.loc[keep_mask].copy()
    # Price is allowed to be missing/zero for non-filled trades, so fill with 0.0
    working["price"] = working["price"].fillna(0.0)
    
    working = working.dropna(subset=["entity_id", "trade_date", "quantity", "desk_type", "structure", "side"])
    working = working[
        (working["entity_id"] != "")
        & (working["desk_type"] != "")
        & (working["structure"] != "")
        & (working["side"] != "")
    ]
    
    # Quantity must be > 0. Price must be > 0 ONLY for primary (filled) trades.
    is_primary = working["status"].fillna("").str.strip().str.casefold().isin(primary_statuses)
    valid_price_mask = (working["price"] > 0) | (~is_primary)
    working = working[(working["quantity"] > 0) & valid_price_mask]

    if working.empty:
        raise RuntimeError("No executable trade rows remained after sanitization.")

    working["proxy_id"] = working["desk_type"] + "_" + working["structure"] + "_" + working["side"]
    working["notional_value"] = working["quantity"] * working["price"]

    trade_days = working["trade_date"].dt.floor("D")
    max_date = trade_days.max()
    working["days_elapsed"] = (max_date - trade_days).dt.days.astype(int)
    working["base_weight"] = np.exp(-TRADE_DECAY_LAMBDA * working["days_elapsed"].astype(float))
    working = working.sort_values(["entity_id", "trade_date", "proxy_id"]).reset_index(drop=True)

    #  1. Assign multipliers based on status tier
    current_status = working["status"].fillna("").str.strip().str.casefold()
    working["status_multiplier"] = np.where(
        current_status.isin(primary_statuses), 1.0,
        np.where(
            current_status.isin(secondary_statuses), 0.5,
            np.where(
                current_status.isin(tertiary_statuses), 0.1,
                0.0
            )
        )
    )
    
    # 2. Discount the interaction weight (so they don't overpower the LightFM embeddings)
    working["base_weight"] = working["base_weight"] * working["status_multiplier"]
    
    # 3. Discount the notional value (so cancelled trades don't inflate Vol Tiers)
    working["notional_value"] = working["notional_value"] * working["status_multiplier"]

    return working, max_date


def build_interactions_df(clean_trade_history: pd.DataFrame) -> pd.DataFrame:
    interactions_df = (
        clean_trade_history.groupby(["entity_id", "proxy_id"], as_index=False)["base_weight"]
        .sum()
        .rename(columns={"base_weight": "cumulative_weight"})
    )
    return interactions_df.sort_values(["entity_id", "proxy_id"]).reset_index(drop=True)


def build_user_feature_frames(clean_trade_history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_notional = clean_trade_history.groupby("entity_id", as_index=False)["notional_value"].sum()
    user_trade_count = clean_trade_history.groupby("entity_id", as_index=False)["status_multiplier"].sum().rename(columns={"status_multiplier": "trade_count"})


    user_summary_df = user_notional.merge(user_trade_count, on="entity_id", how="outer")
    user_summary_df["notional_value"] = user_summary_df["notional_value"].fillna(0.0)
    user_summary_df["trade_count"] = user_summary_df["trade_count"].fillna(0).astype(int)
    user_summary_df["vol_tier"] = safe_quantile_labels(user_summary_df["notional_value"], VOLUME_LABELS)
    user_summary_df["velocity_tier"] = safe_quantile_labels(user_summary_df["trade_count"], VELOCITY_LABELS)
    user_summary_df = user_summary_df.sort_values("entity_id").reset_index(drop=True)

    total_trades_df = user_summary_df[["entity_id", "trade_count"]].rename(columns={"trade_count": "total_trades"})

    desk_distribution_df = (
        clean_trade_history.groupby(["entity_id", "desk_type"], as_index=False)["status_multiplier"]
        .sum()
        .rename(columns={"status_multiplier": "desk_trade_count"})
        .merge(total_trades_df, on="entity_id", how="left")
    )
    desk_distribution_df["desk_weight"] = desk_distribution_df["desk_trade_count"] / desk_distribution_df["total_trades"]
    desk_distribution_df["feature"] = "desk_" + desk_distribution_df["desk_type"]
    desk_wide_df = (
        desk_distribution_df.pivot_table(
            index="entity_id", columns="feature", values="desk_weight", aggfunc="sum", fill_value=0.0
        )
        .reset_index()
    )

    structure_distribution_df = (
        clean_trade_history.groupby(["entity_id", "structure"], as_index=False)["status_multiplier"]
        .sum()
        .rename(columns={"status_multiplier": "structure_trade_count"})
        .merge(total_trades_df, on="entity_id", how="left")
    )
    structure_distribution_df["structure_weight"] = (
        structure_distribution_df["structure_trade_count"] / structure_distribution_df["total_trades"]
    )
    structure_distribution_df["feature"] = "struct_" + structure_distribution_df["structure"]
    structure_wide_df = (
        structure_distribution_df.pivot_table(
            index="entity_id", columns="feature", values="structure_weight", aggfunc="sum", fill_value=0.0
        )
        .reset_index()
    )

    desk_columns = [column for column in desk_wide_df.columns if column != "entity_id"]
    structure_columns = [column for column in structure_wide_df.columns if column != "entity_id"]

    user_features_df = (
        user_summary_df[["entity_id", "vol_tier", "velocity_tier"]]
        .merge(desk_wide_df, on="entity_id", how="left")
        .merge(structure_wide_df, on="entity_id", how="left")
        .sort_values("entity_id")
        .reset_index(drop=True)
    )

    numeric_columns = desk_columns + structure_columns
    if numeric_columns:
        user_features_df[numeric_columns] = user_features_df[numeric_columns].fillna(0.0)

    return user_summary_df, user_features_df


def build_item_features_df(clean_trade_history: pd.DataFrame) -> pd.DataFrame:
    item_features_df = clean_trade_history[["proxy_id"]].drop_duplicates().sort_values("proxy_id").reset_index(drop=True)
    proxy_parts = item_features_df["proxy_id"].str.split("_", n=2, expand=True)

    if proxy_parts.shape[1] != 3:
        raise RuntimeError("Proxy ID decomposition failed because a proxy_id did not split into three parts.")

    item_features_df["item_desk"] = "item_desk_" + proxy_parts[0]
    item_features_df["item_struct"] = "item_struct_" + proxy_parts[1]
    item_features_df["item_side"] = "item_side_" + proxy_parts[2]
    return item_features_df.reset_index(drop=True)


def collect_user_feature_records(user_features_df: pd.DataFrame) -> tuple[list[tuple[str, str, float]], set[str]]:
    numeric_columns = [column for column in user_features_df.columns if column not in {"entity_id", "vol_tier", "velocity_tier"}]
    records: list[tuple[str, str, float]] = []
    feature_names: set[str] = set()

    for _, row in user_features_df.iterrows():
        entity_id = row["entity_id"]
        uid_feature = f"uid_{entity_id}"
        feature_names.add(uid_feature)
        records.append((entity_id, uid_feature, 1.0))

        for column in ("vol_tier", "velocity_tier"):
            feature_name = row[column]
            if pd.notna(feature_name) and str(feature_name):
                feature_name = str(feature_name)
                feature_names.add(feature_name)
                records.append((entity_id, feature_name, 1.0))

        for column in numeric_columns:
            feature_value = row[column]
            if pd.notna(feature_value):
                weight = float(feature_value)
                if weight > 0:
                    feature_names.add(column)
                    records.append((entity_id, column, weight))

    return records, feature_names


def collect_item_feature_records(item_features_df: pd.DataFrame) -> tuple[list[tuple[str, str, float]], set[str]]:
    records: list[tuple[str, str, float]] = []
    feature_names: set[str] = set()

    for _, row in item_features_df.iterrows():
        proxy_id = row["proxy_id"]
        pid_feature = f"pid_{proxy_id}"
        feature_names.add(pid_feature)
        records.append((proxy_id, pid_feature, 1.0))

        for column in ("item_desk", "item_struct", "item_side"):
            feature_name = row[column]
            if pd.notna(feature_name) and str(feature_name):
                feature_name = str(feature_name)
                feature_names.add(feature_name)
                records.append((proxy_id, feature_name, 1.0))

    return records, feature_names


def build_sparse_matrix(
    records: list[tuple[str, str, float]],
    row_index_map: dict[str, int],
    feature_index_map: dict[str, int],
) -> csr_matrix:
    if not records:
        return csr_matrix((len(row_index_map), len(feature_index_map)), dtype=np.float64)

    row_indices = [row_index_map[row_key] for row_key, _, _ in records]
    col_indices = [feature_index_map[feature_name] for _, feature_name, _ in records]
    data = [float(weight) for _, _, weight in records]

    return coo_matrix(
        (data, (row_indices, col_indices)), shape=(len(row_index_map), len(feature_index_map)), dtype=np.float64
    ).tocsr()


async def main() -> None:
    print("1. Connecting to PostgreSQL and loading trade_history...")
    raw_trade_history = await load_trade_history()
    print(f"   -> Retrieved {len(raw_trade_history):,} rows from trade_history.")

    print("2. Phase 1: extracting valid execution rows and sanitizing the raw feed...")
    clean_trade_history, max_date = sanitize_trade_history(raw_trade_history)
    print(f"   -> Rows after sanitization: {len(clean_trade_history):,}")
    print(f"   -> Global max_date used for decay: {max_date}")

    print("3. Phase 3: aggregating the interaction edges...")
    interactions_df = build_interactions_df(clean_trade_history)
    print(f"   -> Unique entity/proxy edges: {len(interactions_df):,}")

    print("4. Phase 4: generating user feature frames...")
    user_summary_df, user_features_df = build_user_feature_frames(clean_trade_history)
    print(f"   -> Users with feature rows: {len(user_features_df):,}")

    print("5. Phase 5: generating item feature frames...")
    item_features_df = build_item_features_df(clean_trade_history)
    print(f"   -> Proxy items with feature rows: {len(item_features_df):,}")

    print("6. Phase 6: creating mappings and sparse matrices...")
    entity_index = {entity_id: index for index, entity_id in enumerate(sorted(clean_trade_history["entity_id"].unique()))}
    proxy_index = {proxy_id: index for index, proxy_id in enumerate(sorted(clean_trade_history["proxy_id"].unique()))}

    interaction_records = list(
        zip(interactions_df["entity_id"], interactions_df["proxy_id"], interactions_df["cumulative_weight"])
    )
    user_feature_records, user_feature_names = collect_user_feature_records(user_features_df)
    item_feature_records, item_feature_names = collect_item_feature_records(item_features_df)
    # feature_index = {
    #     feature_name: index for index, feature_name in enumerate(sorted(user_feature_names | item_feature_names))
    # }
    user_feature_index = {name: idx for idx, name in enumerate(sorted(user_feature_names))}
    item_feature_index = {name: idx for idx, name in enumerate(sorted(item_feature_names))}

    interactions_matrix = build_sparse_matrix(interaction_records, entity_index, proxy_index)
    # user_features_matrix = build_sparse_matrix(user_feature_records, entity_index, feature_index)
    # item_features_matrix = build_sparse_matrix(item_feature_records, proxy_index, feature_index)
    user_features_matrix = build_sparse_matrix(user_feature_records, entity_index, user_feature_index)
    item_features_matrix = build_sparse_matrix(item_feature_records, proxy_index, item_feature_index)

    print(f"   -> Entity count: {len(entity_index):,}")
    print(f"   -> Proxy count: {len(proxy_index):,}")
    # print(f"   -> Feature count: {len(feature_index):,}")
    print(f"   -> User feature count: {len(user_feature_index):,}")
    print(f"   -> Item feature count: {len(item_feature_index):,}")
    print(f"   -> Interactions matrix shape: {interactions_matrix.shape}")
    print(f"   -> User feature matrix shape: {user_features_matrix.shape}")
    print(f"   -> Item feature matrix shape: {item_features_matrix.shape}")

    print("7. Saving ETL artifacts...")
    write_pickle(DIR_PATH / "trade_history_sanitized.pkl", clean_trade_history)
    write_pickle(DIR_PATH / "trade_interactions.pkl", interactions_df)
    write_pickle(DIR_PATH / "user_summary.pkl", user_summary_df)
    write_pickle(DIR_PATH / "user_features.pkl", user_features_df)
    write_pickle(DIR_PATH / "item_features.pkl", item_features_df)
    write_pickle(
        DIR_PATH / "proxy_model_artifacts.pkl",
        {
            "max_date": max_date,
            "entity_index": entity_index,
            "proxy_index": proxy_index,
            # "feature_index": feature_index,
            "interactions_matrix": interactions_matrix,
            "user_features_matrix": user_features_matrix,
            "item_features_matrix": item_features_matrix,
        },
    )

    print("SUCCESS: Proxy-model ETL artifacts were written to disk.")


if __name__ == "__main__":
    asyncio.run(main())
