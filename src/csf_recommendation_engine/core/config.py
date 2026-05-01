from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="prod", alias="APP_ENV")
    app_name: str = Field(default="csf-recommendation-engine", alias="APP_NAME")
    app_debug: bool = Field(default=False, alias="APP_DEBUG")
    app_timezone: str = Field(default="America/New_York", alias="APP_TIMEZONE")

    postgres_dsn: str = Field(default="", alias="POSTGRES_DSN")
    postgres_readonly_dsn: str = Field(default="", alias="POSTGRES_READONLY_DSN")

    # azure_storage_connection_string: str = Field(default="", alias="AZURE_STORAGE_CONNECTION_STRING")
    # azure_storage_container: str = Field(default="csf-recommender-data", alias="AZURE_STORAGE_CONTAINER")
    # features_latest_blob_path: str = Field(
    #     default="features/daily_features_latest.parquet", alias="FEATURES_LATEST_BLOB_PATH"
    # )
    # champion_model_prefix: str = Field(default="models/", alias="CHAMPION_MODEL_PREFIX")
    champion_model_path: str = Field(default="src/csf_recommendation_engine/scripts/proxy_model_live.pkl", alias="CHAMPION_MODEL_PATH")
    model_artifacts_path: str = Field(default="src/csf_recommendation_engine/scripts/proxy_model_artifacts.pkl", alias="MODEL_ARTIFACTS_PATH")

    heuristics_artifact_dir: str = Field(default="src/data/heuristics", alias="HEURISTICS_ARTIFACT_DIR")
    heuristics_entity_features_latest_filename: str = Field(
        default="entity_features_latest.parquet", alias="HEURISTICS_ENTITY_FEATURES_LATEST_FILENAME"
    )
    heuristics_instrument_features_latest_filename: str = Field(
        default="instrument_features_latest.parquet",
        alias="HEURISTICS_INSTRUMENT_FEATURES_LATEST_FILENAME",
    )
    heuristics_active_venue_lookback_days: int = Field(
        default=30, alias="HEURISTICS_ACTIVE_VENUE_LOOKBACK_DAYS"
    )
    heuristics_history_days: int = Field(default=365, alias="HEURISTICS_HISTORY_DAYS")
    heuristics_completed_statuses: str = Field(
        default="Filled", alias="HEURISTICS_COMPLETED_STATUSES"
    )

    rerank_enabled: bool = Field(default=True, alias="RERANK_ENABLED")
    rerank_require_heuristics: bool = Field(default=False, alias="RERANK_REQUIRE_HEURISTICS")
    rerank_candidate_pool_size: int = Field(default=200, alias="RERANK_CANDIDATE_POOL_SIZE")
    rerank_weight_lightfm: float = Field(default=1.0, alias="RERANK_WEIGHT_LIGHTFM")
    rerank_weight_time_affinity: float = Field(default=0.1, alias="RERANK_WEIGHT_TIME_AFFINITY")
    rerank_weight_recency: float = Field(default=0.1, alias="RERANK_WEIGHT_RECENCY")
    rerank_weight_size_fit: float = Field(default=0.1, alias="RERANK_WEIGHT_SIZE_FIT")
    rerank_recency_halflife_days: float = Field(
        default=30.0, alias="RERANK_RECENCY_HALFLIFE_DAYS"
    )

    nightly_pipeline_enabled: bool = Field(default=True, alias="NIGHTLY_PIPELINE_ENABLED")
    nightly_schedule_cron: str = Field(default="0 2 * * *", alias="NIGHTLY_SCHEDULE_CRON")
    nightly_schedule_timezone: str = Field(
        default="America/New_York", alias="NIGHTLY_SCHEDULE_TIMEZONE"
    )
    nightly_min_working_quantity: int = Field(default=5, alias="NIGHTLY_MIN_WORKING_QUANTITY")
    nightly_top_k: int = Field(default=3, alias="NIGHTLY_TOP_K")
    nightly_batch_size: int = Field(default=25, alias="NIGHTLY_BATCH_SIZE")
    nightly_max_concurrency: int = Field(default=2, alias="NIGHTLY_MAX_CONCURRENCY")
    cross_match_threshold: float = Field(default=0.6, alias="CROSS_MATCH_THRESHOLD")
    rec_refresh_enabled: bool = Field(default=True, alias="REC_REFRESH_ENABLED")

    # top_k: int = Field(default=5, alias="TOP_K")
    # max_candidates: int = Field(default=5000, alias="MAX_CANDIDATES")
    # recommend_timeout_seconds: float = Field(default=1.5, alias="RECOMMEND_TIMEOUT_SECONDS")
    # candidate_universe_policy: str = Field(default="", alias="CANDIDATE_UNIVERSE_POLICY")
    extraction_min_rows: int = Field(default=1, alias="EXTRACTION_MIN_ROWS")
    extraction_null_threshold: float = Field(default=0.05, alias="EXTRACTION_NULL_THRESHOLD")


    # Reserved placeholders for future service bus settings. Runtime integration is deferred.
    # service_bus_namespace: str = Field(default="", alias="SERVICE_BUS_NAMESPACE")
    # service_bus_topic: str = Field(default="", alias="SERVICE_BUS_TOPIC")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
