"""Inference Engine for the Internal Crossing Proxy Model."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightfm import LightFM

DIR_PATH = Path(__file__).parent


def load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


class CrossingEngine:
    def __init__(self, model_path: Path):
        print("Initializing Crossing Engine...")
        try:
            payload = load_pickle(model_path)
            self.model: LightFM = payload["model"]
            self.entity_index: dict[str, int] = payload["entity_index"]
            self.proxy_index: dict[str, int] = payload["proxy_index"]
            self.user_feature_index: dict[str, int] = payload.get("user_feature_index", {})
            self.item_feature_index: dict[str, int] = payload.get("item_feature_index", {})
            
            # Load the raw user features matrix (required for LightFM prediction)
            artifacts = load_pickle(DIR_PATH / "proxy_model_artifacts.pkl")
            self.user_features_matrix = artifacts["user_features_matrix"]
            self.item_features_matrix = artifacts["item_features_matrix"]
            
            # Create a reverse lookup for entities (Integer ID -> UUID String)
            self.reverse_entity_index = {v: k for k, v in self.entity_index.items()}
            print(f" -> Engine ready. Loaded {len(self.entity_index)} clients and {len(self.proxy_index)} intent profiles.")
        except FileNotFoundError as e:
            raise RuntimeError(f"Model artifacts missing. Ensure training script has run. {e}")


    def get_mirror_proxy(self, desk: str, structure: str, side: str) -> str:
        """Flips the incoming order side to find the Natural Counterparty intent."""
        side = side.upper().strip()
        if side == "BUY":
            target_side = "SELL"
        elif side == "SELL":
            target_side = "BUY"
        else:
            raise ValueError(f"Invalid side '{side}'. Must be 'BUY' or 'SELL'.")
            
        # Matches the normalization logic from the ETL phase
        desk = desk.upper().strip().replace(" ", "-")
        structure = structure.upper().strip().replace(" ", "-")
        
        return f"{desk}_{structure}_{target_side}"


    def recommend_counterparties(
        self, 
        requesting_entity_id: str, 
        target_desk: str, 
        target_structure: str, 
        target_side: str, 
        top_k: int = 5
    ) -> pd.DataFrame:
        """
        Executes Stage 1: Latent Behavioral Ranking.
        Given an incoming block order, ranks all clients by their affinity for the OPPOSITE side.
        """
        # 1. The Mirror Pivot
        target_proxy = self.get_mirror_proxy(target_desk, target_structure, target_side)
        
        if target_proxy not in self.proxy_index:
            raise ValueError(f"The intent profile '{target_proxy}' has never been observed in training data. Cannot compute affinity.")
            
        target_proxy_int = self.proxy_index[target_proxy]
        
        # 2. Prepare the prediction arrays
        # We want to score EVERY client in the database against this single target proxy
        n_users = len(self.entity_index)
        user_ids_array = np.arange(n_users)
        item_ids_array = np.full(n_users, target_proxy_int) # Fill array with the single target item ID
        
        # 3. Model Inference
        # We pass the user and item feature matrices so the model utilizes the float-encoded distributions
        scores = self.model.predict(
            user_ids=user_ids_array,
            item_ids=item_ids_array,
            user_features=self.user_features_matrix,
            item_features=self.item_features_matrix
        )
        
        # 4. Map scores back to UUIDs
        results = []
        for internal_id, score in zip(user_ids_array, scores):
            entity_uuid = self.reverse_entity_index[internal_id]
            results.append({
                "candidate_entity_id": entity_uuid,
                "affinity_score": float(score)
            })
            
        df_results = pd.DataFrame(results)
        
        # 5. The Exclusion Filter (Don't recommend the client to themselves)
        df_results = df_results[df_results["candidate_entity_id"] != requesting_entity_id]
        
        # 6. Sort by highest affinity
        df_results = df_results.sort_values(by="affinity_score", ascending=False).reset_index(drop=True)
        
        return df_results.head(top_k)


def stage_2_rules_engine_mock(
    candidates_df: pd.DataFrame, 
    specific_instrument: str, 
    live_inventory_db: dict
) -> pd.DataFrame:
    """
    Executes Stage 2: Deterministic Reality Check.
    In a real system, this queries the `exposure_summary` or `voice_inquiries` tables
    to see if the top latent matches are actually holding the specific CUSIP/Contract today.
    """
    # Create a boolean mask indicating if the candidate holds the specific instrument
    has_inventory = candidates_df["candidate_entity_id"].map(
        lambda x: specific_instrument in live_inventory_db.get(x, [])
    )
    
    candidates_df["live_inventory_match"] = has_inventory
    
    # Push clients with actual live inventory to the absolute top, regardless of latent score
    candidates_df = candidates_df.sort_values(
        by=["live_inventory_match", "affinity_score"], 
        ascending=[False, False]
    ).reset_index(drop=True)
    
    return candidates_df


def main():
    # 1. Initialize the Engine
    engine = CrossingEngine(DIR_PATH / "proxy_model_live.pkl")
    
    # 2. Simulate an incoming block trade call from a Hedge Fund
    incoming_client = "727a24cc-aa0c-4876-b8c6-b4fdc784ba47" # Replace with a real UUID from your DB
    desk = "CRUDE"
    structure = "FLAT PRICE"
    side = "BUY"
    specific_contract = "WTI_DEC26_JAN27_CALSPREAD"
    
    print(f"\n--- INCOMING ORDER ---")
    print(f"Client: {incoming_client} wants to {side} a {desk} {structure}.")
    print(f"Target Proxy to Query: {engine.get_mirror_proxy(desk, structure, side)}\n")
    
    # 3. Execute Stage 1 (Latent Recommendation)
    print("Executing Stage 1: Latent Behavioral Search...")
    stage_1_candidates = engine.recommend_counterparties(
        requesting_entity_id=incoming_client,
        target_desk=desk,
        target_structure=structure,
        target_side=side,
        top_k=5
    )
    print(stage_1_candidates.to_string(index=False))
    
    # 4. Execute Stage 2 (Deterministic Filtering)
    # MOCK DATA: Simulating what your exposure tables would tell you today
    print(f"\nExecuting Stage 2: Real-time Inventory Check for {specific_contract}...")
    mock_live_inventory = {
        # Let's pretend the 3rd ranked latent candidate actually holds this exact contract today
        stage_1_candidates.iloc[2]["candidate_entity_id"]: [specific_contract, "OTHER_ASSET"]
    }
    
    final_broker_list = stage_2_rules_engine_mock(
        stage_1_candidates.copy(), 
        specific_contract, 
        mock_live_inventory
    )
    
    print(final_broker_list.to_string(index=False))
    print("\nACTION: Broker should call the top candidate immediately. They are structurally aligned AND hold the specific inventory.")

if __name__ == "__main__":
    main()