"""Nightly extraction job for CSF recommendation engine.

Specifically for shadow LightFM model training.
Currently not implemented.
"""

import asyncio
from datetime import datetime, timezone
import logging

import pandas as pd

from csf_recommendation_engine.core.config import get_settings

logger = logging.getLogger(__name__)


async def fetch_raw_feature_frame(_readonly_dsn: str) -> pd.DataFrame:
    # IMPORTANT TODO CHECKPOINT:
    # Source tables and SQL scope must be finalized before query implementation.
    raise NotImplementedError(
        "Nightly extraction SQL is blocked pending source-table scope decision."
    )


async def run_nightly_extraction() -> None:
    settings = get_settings()
    # if not settings.postgres_readonly_dsn:
    #     raise RuntimeError("POSTGRES_READONLY_DSN is required for nightly extraction")
    # if not settings.azure_storage_connection_string:
    #     raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING is required for nightly extraction")

    # raw_df = await fetch_raw_feature_frame(settings.postgres_readonly_dsn)
    # transformed_df = apply_feature_transformations(raw_df)
    # validate_feature_frame(
    #     transformed_df,
    #     min_rows=settings.extraction_min_rows,
    #     null_threshold=settings.extraction_null_threshold,
    # )

    # payload = to_parquet_bytes(transformed_df)
    # run_time_utc = datetime.now(timezone.utc)
    # dated_blob_path, latest_blob_path = build_feature_blob_paths(
    #     latest_blob_path=settings.features_latest_blob_path,
    #     run_time_utc=run_time_utc,
    # )

    # blob = BlobStore(
    #     connection_string=settings.azure_storage_connection_string,
    #     container=settings.azure_storage_container,
    # )
    # await blob.upload_blob_bytes(dated_blob_path, payload, overwrite=True)
    # await blob.upload_blob_bytes(latest_blob_path, payload, overwrite=True)
    # logger.info(
    #     "Nightly features uploaded",
    #     extra={
    #         "dated_blob_path": dated_blob_path,
    #         "latest_blob_path": latest_blob_path,
    #         "row_count": len(transformed_df),
    #     },
    # )


if __name__ == "__main__":
    asyncio.run(run_nightly_extraction())
