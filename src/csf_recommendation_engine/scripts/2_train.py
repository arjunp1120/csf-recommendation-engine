"""Standalone Training script for the LightFM Proxy Model in a Small-Data regime."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.evaluation import auc_score, precision_at_k
from scipy.sparse import coo_matrix, csr_matrix

DIR_PATH = Path(__file__).parent


def load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_pickle(path: Path, payload: object) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def build_sparse_matrix(
    df_agg: pd.DataFrame, 
    row_index_map: dict[str, int], 
    col_index_map: dict[str, int]
) -> csr_matrix:
    """Safely constructs a sparse matrix matching the exact dimensions of our global indices."""
    if df_agg.empty:
        return csr_matrix((len(row_index_map), len(col_index_map)), dtype=np.float64)

    row_indices = []
    col_indices = []
    data = []

    for _, row in df_agg.iterrows():
        # Only map entities and proxies that exist in our global index
        if row["entity_id"] in row_index_map and row["proxy_id"] in col_index_map:
            row_indices.append(row_index_map[row["entity_id"]])
            col_indices.append(col_index_map[row["proxy_id"]])
            data.append(float(row["cumulative_weight"]))

    return coo_matrix(
        (data, (row_indices, col_indices)), 
        shape=(len(row_index_map), len(col_index_map)), 
        dtype=np.float64
    ).tocsr()


def chronological_train_test_split(
    clean_trade_history: pd.DataFrame, 
    entity_index: dict[str, int], 
    proxy_index: dict[str, int], 
    test_ratio: float = 0.15
) -> tuple[csr_matrix, csr_matrix]:
    """Splits raw chronological trades to prevent data leakage, then builds aggregated matrices."""
    print(f" -> Splitting {len(clean_trade_history)} raw interactions chronologically...")
    
    # Sort chronologically (oldest to newest)
    df = clean_trade_history.sort_values(["entity_id", "trade_date"], ascending=[True, True])
    
    train_records = []
    test_records = []
    
    # Split chronologically per client
    for _, group in df.groupby("entity_id"):
        n_total = len(group)
        # For clients with very few trades, leave them entirely in train to avoid cold-start tests
        n_test = max(1, int(n_total * test_ratio)) if n_total > 2 else 0
        
        if n_test > 0:
            train_records.append(group.iloc[:-n_test])
            test_records.append(group.iloc[-n_test:])
        else:
            train_records.append(group)
            
    train_df = pd.concat(train_records) if train_records else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_records) if test_records else pd.DataFrame(columns=df.columns)
    
    # Aggregate weights just like Phase 3 of the ETL
    train_agg = train_df.groupby(["entity_id", "proxy_id"], as_index=False)["base_weight"].sum()
    train_agg.rename(columns={"base_weight": "cumulative_weight"}, inplace=True)
    
    test_agg = test_df.groupby(["entity_id", "proxy_id"], as_index=False)["base_weight"].sum()
    test_agg.rename(columns={"base_weight": "cumulative_weight"}, inplace=True)
    
    train_matrix = build_sparse_matrix(train_agg, entity_index, proxy_index)
    test_matrix = build_sparse_matrix(test_agg, entity_index, proxy_index)
    
    return train_matrix, test_matrix


def main() -> None:
    print("1. Loading ETL Artifacts...")
    try:
        artifacts = load_pickle(DIR_PATH / "proxy_model_artifacts.pkl")
        clean_trade_history = load_pickle(DIR_PATH / "trade_history_sanitized.pkl")
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Run the ETL script first. {e}")
        return

    entity_index = artifacts["entity_index"]
    proxy_index = artifacts["proxy_index"]
    user_features_matrix = artifacts["user_features_matrix"]
    item_features_matrix = artifacts["item_features_matrix"]

    print("2. Constructing Time-Aware Graph Splits...")
    train_interactions, test_interactions = chronological_train_test_split(
        clean_trade_history, entity_index, proxy_index, test_ratio=0.15
    )
    
    print(f"    Train Matrix Density: {train_interactions.nnz} edges")
    print(f"    Test Matrix Density: {test_interactions.nnz} edges")

    if test_interactions.nnz == 0:
        print("WARNING: Dataset is too small to generate a valid test holdout. Evaluation metrics will be skipped.")

    print("\n3. Initializing LightFM (Small Data Architecture)...")
    # HYPERPARAMETER RATIONALE:
    # - no_components=16: Forces aggressive dimensionality reduction to prevent memorizing 25 clients.
    # - user_alpha/item_alpha=1e-4: High L2 regularization penalty to combat overfitting on small data.
    # - loss='warp': Optimizes for the top of the recommendation ranking list.
    model = LightFM(
        no_components=16,
        loss="warp",
        learning_rate=0.02,
        user_alpha=1e-4,
        item_alpha=1e-4,
        random_state=42
    )

    num_threads = os.cpu_count() or 1
    epochs = 40

    print(f"4. Fitting Model (Epochs: {epochs}, Threads: {num_threads})...")
    model.fit(
        train_interactions,
        user_features=user_features_matrix,
        item_features=item_features_matrix,
        epochs=epochs,
        num_threads=num_threads,
        verbose=True
    )

    if test_interactions.nnz > 0:
        print("\n5. Running Objective Evaluation...")
        # Note: We omit `train_interactions` in the evaluation calls so the metric does not penalize 
        # the model for recommending a proxy the client has traded before (desired behavior in OTC).
        
        train_auc = auc_score(
            model, train_interactions, user_features=user_features_matrix, item_features=item_features_matrix, num_threads=num_threads
        ).mean()
        test_auc = auc_score(
            model, test_interactions, user_features=user_features_matrix, item_features=item_features_matrix, num_threads=num_threads
        ).mean()

        train_p3 = precision_at_k(
            model, train_interactions, k=3, user_features=user_features_matrix, item_features=item_features_matrix, num_threads=num_threads
        ).mean()
        test_p3 = precision_at_k(
            model, test_interactions, k=3, user_features=user_features_matrix, item_features=item_features_matrix, num_threads=num_threads
        ).mean()

        print(f" -> Train AUC:         {train_auc:.4f}")
        print(f" -> Test AUC:          {test_auc:.4f}  (>0.50 is better than random)")
        print(f" -> Train Precision@3: {train_p3:.4f}")
        print(f" -> Test Precision@3:  {test_p3:.4f}")

    print("\n6. Exporting Trained Live Model...")
    live_payload = {
        "model": model,
        "entity_index": entity_index,
        "proxy_index": proxy_index,
        "user_feature_index": artifacts.get("user_feature_index", {}),
        "item_feature_index": artifacts.get("item_feature_index", {}),
    }
    write_pickle(DIR_PATH / "proxy_model_live.pkl", live_payload)
    print("SUCCESS: `proxy_model_live.pkl` written to disk. Ready for inference API.")


if __name__ == "__main__":
    main()