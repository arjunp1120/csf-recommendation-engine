import asyncio
import asyncpg
import pandas as pd
import numpy as np
import re
import pickle
from rapidfuzz import process, fuzz
from lightfm.data import Dataset
from lightfm import LightFM
from lightfm.evaluation import auc_score
import ssl

# ==========================================
# 0. DATABASE CONFIGURATION
# ==========================================
# Replace with your actual Azure PostgreSQL credentials
# DB_DSN = "postgresql://username:password@your-server.postgres.database.azure.com:5432/your_database"
from os import getenv
DB_DSN = "postgresql://csf_app_admin:NapuMyRdQkjqP%24asqFzi9dmES%40QN0u%40y@pgdbcsf01.postgres.database.azure.com:5432/csfeu2dpgdb01"

async def extract_data():
    """Asynchronously pull data from Postgres and return Pandas DataFrames."""
    print("1. Connecting to PostgreSQL via asyncpg...")
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    conn = await asyncpg.connect(dsn=DB_DSN, ssl=ssl_ctx)
    
    try:
        # Fetch Clients
        print("   -> Fetching client entities and profiles...")
        clients_query = """
            SELECT ce.entity_id, ce.short_code, ce.entity_name, cp.avg_monthly_volume, cp.preferred_structure
            FROM client_entities ce
            LEFT JOIN client_profiles cp ON ce.entity_id = cp.entity_id
            WHERE ce.is_active = true;
        """
        clients_records = await conn.fetch(clients_query)
        df_clients = pd.DataFrame([dict(r) for r in clients_records])
        
        # Fetch Trades
        print("   -> Fetching completed trades...")
        trades_query = """
            SELECT trade_id, entity_id, contra_party, status
            FROM trades
            WHERE status IN ('Filled', 'Done for Day', 'Calculated');
        """
        trades_records = await conn.fetch(trades_query)
        df_trades = pd.DataFrame([dict(r) for r in trades_records])
        
        # Fetch Inquiries (Axes)
        print("   -> Fetching voice inquiries (Axes)...")
        axes_query = """
            SELECT entity_id, side, product, quantity
            FROM voice_inquiries
            WHERE inquiry_timestamp > NOW() - INTERVAL '30 DAYS';
        """
        axes_records = await conn.fetch(axes_query)
        df_axes = pd.DataFrame([dict(r) for r in axes_records])
        
    finally:
        await conn.close()
        
    return df_clients, df_trades, df_axes

def cleanse_string(text):
    """Sanitize strings for fuzzy matching."""
    if pd.isna(text): return ""
    text = str(text).upper().strip()
    text = re.sub(r'\b(LLC|INC|CORP|LTD|LP|PLC|GROUP)\b', '', text)
    return re.sub(r'[^\w\s]', '', text).strip()

def main():
    # ==========================================
    # 1. LOAD ASYNC DATA
    # ==========================================
    # Run the async loop to grab our dataframes
    df_clients, df_trades, df_axes = asyncio.run(extract_data())
    
    # ==========================================
    # 2. RESOLVE CONTRA_PARTY TO UUIDs
    # ==========================================
    print("2. Resolving text contra_parties to UUIDs...")
    df_trades['clean_contra'] = df_trades['contra_party'].apply(cleanse_string)
    df_clients['clean_name'] = df_clients['entity_name'].apply(cleanse_string)
    df_clients['clean_short'] = df_clients['short_code'].apply(cleanse_string)

    name_to_uuid = dict(zip(df_clients['clean_name'], df_clients['entity_id']))
    short_to_uuid = dict(zip(df_clients['clean_short'], df_clients['entity_id']))
    client_names_list = list(name_to_uuid.keys())

    resolved_uuids = []
    for contra in df_trades['clean_contra']:
        if not contra:
            resolved_uuids.append(None)
            continue
        if contra in short_to_uuid:
            resolved_uuids.append(short_to_uuid[contra])
        elif contra in name_to_uuid:
            resolved_uuids.append(name_to_uuid[contra])
        else:
            match = process.extractOne(contra, client_names_list, scorer=fuzz.WRatio)
            if match and match[1] >= 85:
                resolved_uuids.append(name_to_uuid[match[0]])
            else:
                resolved_uuids.append(None)

    df_trades['contra_id'] = resolved_uuids
    valid_trades = df_trades.dropna(subset=['contra_id'])

    # ==========================================
    # 3. BUCKETIZE CONTINUOUS FEATURES
    # ==========================================
    print("3. Bucketizing features...")
    df_clients['avg_monthly_volume'] = df_clients['avg_monthly_volume'].astype(float).fillna(0)
    df_clients['vol_tier'] = pd.qcut(df_clients['avg_monthly_volume'], q=4, labels=['vol_Q1', 'vol_Q2', 'vol_Q3', 'vol_Q4'], duplicates='drop').astype(str)
    df_clients['preferred_structure'] = df_clients['preferred_structure'].fillna('struct_UNKNOWN')

    # ==========================================
    # 4. BUILD THE MATRICES
    # ==========================================
    print("4. Fitting Dataset and building SciPy matrices...")
    dataset = Dataset()

    all_entity_ids = df_clients['entity_id'].unique().tolist()
    user_features_vocab = [f"vol_Q{i}" for i in range(1, 5)] + df_clients['preferred_structure'].unique().tolist()
    item_features_vocab = ['axis_BUY', 'axis_SELL', 'axis_NEUTRAL']

    dataset.fit(
        users=all_entity_ids,
        items=all_entity_ids,
        user_features=user_features_vocab,
        item_features=item_features_vocab
    )

    interactions_data = list(zip(valid_trades['entity_id'], valid_trades['contra_id'], [1] * len(valid_trades)))
    (interactions_matrix, weights_matrix) = dataset.build_interactions(interactions_data)

    user_features_data = [(row['entity_id'], [row['vol_tier'], row['preferred_structure']]) for _, row in df_clients.iterrows()]
    user_features_matrix = dataset.build_user_features(user_features_data)

    axis_mapping = df_axes.groupby('entity_id')['side'].apply(lambda sides: [f"axis_{side.upper()}" for side in sides]).to_dict()
    item_features_data = [(eid, axis_mapping.get(eid, ['axis_NEUTRAL'])) for eid in all_entity_ids]
    item_features_matrix = dataset.build_item_features(item_features_data)

    # ==========================================
    # 5. TRAIN THE CORE ENGINE
    # ==========================================
    print("5. Training Champion Model (WARP Loss, 64 Components)...")
    model = LightFM(
        no_components=64,
        loss='warp',
        learning_rate=0.05,
        random_state=42
    )

    model.fit(
        interactions=interactions_matrix,
        user_features=user_features_matrix,
        item_features=item_features_matrix,
        epochs=30,
        num_threads=4  # Uses your local CPU cores
    )

    # ==========================================
    # 6. VALIDATION & EXPORT
    # ==========================================
    print("6. Evaluating Accuracy...")
    train_auc = auc_score(
        model, 
        interactions_matrix, 
        user_features=user_features_matrix, 
        item_features=item_features_matrix, 
        num_threads=4
    ).mean()

    print(f"Final AUC Score: {train_auc:.4f}")

    production_payload = {
        'model': model,
        'dataset': dataset,
        'item_features_matrix': item_features_matrix
    }

    with open('champion_model_v1.pkl', 'wb') as f:
        pickle.dump(production_payload, f)

    print("SUCCESS: `champion_model_v1.pkl` generated.")

if __name__ == "__main__":
    # If you run into "Event loop is already running" issues in a Jupyter Notebook, 
    # you may need to use `await extract_data()` instead of `asyncio.run()`
    main()