import pandas as pd
import numpy as np
import sys
from pathlib import Path

DIR_PATH = Path(__file__).parent

def load_data():
    try:
        df_clients = pd.read_pickle(DIR_PATH / 'raw_clients.pkl')
        df_trades = pd.read_pickle(DIR_PATH / 'raw_trades.pkl')
        df_axes = pd.read_pickle(DIR_PATH / 'raw_axes.pkl')
        return df_clients, df_trades, df_axes
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Data payload not found. {e}")
        sys.exit(1)

def separator(title):
    print(f"\n{'='*60}")
    print(f" {title.upper()} ")
    print(f"{'='*60}")

def analyze_clients(df):
    separator("Client Entity Diagnostic (raw_clients.pkl)")
    
    print(f"Total Client Records: {len(df):,}")
    print(f"Total Features: {df.shape[1]}")
    
    print("\n--- Feature Missingness ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing_Count': missing, 'Missing_Percentage': missing_pct})
    print(missing_df[missing_df['Missing_Count'] > 0].to_string() if not missing_df[missing_df['Missing_Count'] > 0].empty else "Zero missing values detected.")

    print("\n--- Categorical Cardinality ---")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if col not in ['entity_id', 'entity_name', 'short_code']: # Ignore pure IDs
            unique_count = df[col].nunique()
            print(f"{col}: {unique_count} unique categorical states.")
            if unique_count < 15:
                print(df[col].value_counts(normalize=True).apply(lambda x: f"  > {x:.2%}").to_string())

    print("\n--- Numerical Distribution ---")
    num_cols = df.select_dtypes(include=[np.number]).columns
    if not num_cols.empty:
        print(df[num_cols].describe(percentiles=[.25, .5, .75, .90, .99]).T.to_string())
    else:
        print("No pure numerical arrays detected.")

def analyze_trades(df_trades, df_clients):
    separator("Transaction Diagnostic (raw_trades.pkl)")
    
    print(f"Total Historical Trades: {len(df_trades):,}")
    
    print("\n--- Data Integrity ---")
    # Identify how many trades map to actual known clients
    known_clients = set(df_clients['entity_id'].unique())
    trades_with_known_clients = df_trades['entity_id'].isin(known_clients).sum()
    print(f"Trades mapped to known client UUIDs: {trades_with_known_clients:,} ({(trades_with_known_clients/len(df_trades)):.2%})")
    
    missing_contra = df_trades['contra_party'].isnull().sum()
    print(f"Trades with NULL contra_party strings: {missing_contra:,} ({(missing_contra/len(df_trades)):.2%})")

    print("\n--- Interaction Density ---")
    unique_active_clients = df_trades['entity_id'].nunique()
    unique_counterparties = df_trades['contra_party'].nunique()
    
    print(f"Unique Active Clients: {unique_active_clients:,}")
    print(f"Unique Distinct Counterparties: {unique_counterparties:,}")
    
    # Calculate matrix sparsity
    total_possible_edges = unique_active_clients * unique_counterparties
    actual_edges = len(df_trades[['entity_id', 'contra_party']].drop_duplicates())
    if total_possible_edges > 0:
        sparsity = 1.0 - (actual_edges / total_possible_edges)
        print(f"Bipartite Graph Sparsity: {sparsity:.6%} (Higher = harder for ML to solve)")

    print("\n--- Concentration Metrics (Loyalty) ---")
    trades_per_client = df_trades.groupby('entity_id').size()
    print(f"Average trades per active client: {trades_per_client.mean():.2f}")
    print(f"Median trades per active client: {trades_per_client.median():.2f}")
    print(f"Max trades by a single client: {trades_per_client.max():,}")
    
    # Counterparty monopolization
    top_dealers = df_trades['contra_party'].value_counts(normalize=True).head(5)
    print("\nTop 5 Counterparties by Global Market Share:")
    print(top_dealers.apply(lambda x: f"{x:.2%}").to_string())

def analyze_axes(df_axes, df_clients):
    separator("Liquidity Axis Diagnostic (raw_axes.pkl)")
    
    print(f"Total Active Axes: {len(df_axes):,}")
    
    if df_axes.empty:
        print("Axis dataframe is empty.")
        return

    print("\n--- Axis Distribution ---")
    if 'side' in df_axes.columns:
        sides = df_axes['side'].value_counts(normalize=True)
        print(sides.apply(lambda x: f"  > {x:.2%}").to_string())
    
    print("\n--- Axis Saturation ---")
    known_clients = set(df_clients['entity_id'].unique())
    axes_with_known_clients = df_axes['entity_id'].isin(known_clients).sum()
    clients_with_axes = df_axes['entity_id'].nunique()
    total_clients = len(df_clients)
    
    print(f"Axes mapped to known client UUIDs: {axes_with_known_clients:,} ({(axes_with_known_clients/len(df_axes)):.2%})")
    print(f"Clients currently holding active axes: {clients_with_axes:,} / {total_clients:,} ({(clients_with_axes/total_clients):.2%})")

def main():
    print("Initializing Data Diagnostics...")
    df_clients, df_trades, df_axes = load_data()
    
    analyze_clients(df_clients)
    analyze_trades(df_trades, df_clients)
    analyze_axes(df_axes, df_clients)
    
    separator("DIAGNOSTIC COMPLETE")

if __name__ == "__main__":
    main()