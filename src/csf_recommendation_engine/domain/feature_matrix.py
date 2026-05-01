from io import BytesIO

import pandas as pd
from scipy import sparse


def build_item_feature_matrix(parquet_bytes: BytesIO) -> sparse.csr_matrix:
    feature_df = pd.read_parquet(parquet_bytes)

    if feature_df.empty:
        return sparse.csr_matrix((0, 0))

    encoded = pd.get_dummies(feature_df, dummy_na=True)
    return sparse.csr_matrix(encoded.to_numpy(dtype=float))
