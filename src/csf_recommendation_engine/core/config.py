from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="prod", alias="APP_ENV")
    app_name: str = Field(default="csf-recommendation-engine", alias="APP_NAME")
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")
    app_debug: bool = Field(default=False, alias="APP_DEBUG")
    app_timezone: str = Field(default="America/New_York", alias="APP_TIMEZONE")

    database_url: str = Field(default="", alias="DATABASE_URL")
    # postgres_readonly_dsn: str = Field(default="", alias="POSTGRES_READONLY_DSN")

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
    rec_refresh_enabled: bool = Field(default=False, alias="REC_REFRESH_ENABLED")

    # LLM / DAF SDK settings
    daf_base_url: str = Field(default="https://daf-sdk-backend.azurewebsites.net", alias="DAF_BASE_URL")
    daf_api_key: str = Field(default="", alias="DAF_API_KEY")
    daf_agent_direct_api_key: str = Field(default="", alias="DAF_AGENT_DIRECT_API_KEY")
    recommendation_agent_id: str = Field(default="", alias="RECOMMENDATION_AGENT_ID")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_model_name: str = Field(default="gpt-4o", alias="LLM_MODEL_NAME")
    llm_model_provider: str = Field(default="OpenAI", alias="LLM_MODEL_PROVIDER")
    llm_temperature: str = Field(default="0.4", alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=4096, alias="LLM_MAX_TOKENS")
    llm_enabled: bool = Field(default=True, alias="LLM_ENABLED")
    llm_candidate_pool_size: int = Field(default=10, alias="LLM_CANDIDATE_POOL_SIZE")
    llm_request_timeout_s: float = Field(default=60.0, alias="LLM_REQUEST_TIMEOUT_S")

    # Externally configured swarm/agent IDs on the DAF platform.
    # Empty default = swarm not yet configured; engine will skip (or raise) at call time.
    daf_tagger_agent_id: str = Field(default="b78ee6e0-9cd8-40ea-98bc-77bf442f60e9", alias="DAF_TAGGER_AGENT_ID")
    daf_profiler_swarm_id: str = Field(default="", alias="DAF_PROFILER_SWARM_ID")
    daf_market_reader_swarm_id: str = Field(default="", alias="DAF_MARKET_READER_SWARM_ID")
    daf_recommender_explainer_swarm_id: str = Field(default="", alias="DAF_RECOMMENDER_EXPLAINER_SWARM_ID")
    daf_match_strategist_swarm_id: str = Field(default="", alias="DAF_MATCH_STRATEGIST_SWARM_ID")
    daf_coverage_coach_swarm_id: str = Field(default="", alias="DAF_COVERAGE_COACH_SWARM_ID")
    # Agent (not swarm) used by scripts/seed_instrument_products.py to resolve
    # long-tail trade_history.instrument_name -> canonical product (plan Step 0.10).
    daf_instrument_resolver_agent_id: str = Field(
        default="", alias="DAF_INSTRUMENT_RESOLVER_AGENT_ID"
    )

    # Embedding provider settings: DEFERRED for v1 (pgvector unavailable on local DB; see plan §17).
    # When enabled, add: EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_DIMENSION.

    # Critic / validator settings (plan §11.4)
    critic_mode: str = Field(default="shadow", alias="CRITIC_MODE")  # 'shadow' | 'strict'
    critic_promotion_violation_rate: float = Field(default=0.01, alias="CRITIC_PROMOTION_VIOLATION_RATE")
    critic_promotion_window: int = Field(default=500, alias="CRITIC_PROMOTION_WINDOW")

    # Voice inquiry / Match Universe Cache settings (plan §10, §17)
    voice_inquiry_expiry_minutes: int = Field(default=240, alias="VOICE_INQUIRY_EXPIRY_MINUTES")
    match_universe_rebuild_interval_seconds: int = Field(
        default=300, alias="MATCH_UNIVERSE_REBUILD_INTERVAL_SECONDS"
    )
    match_qty_overlap_ratio: float = Field(default=0.5, alias="MATCH_QTY_OVERLAP_RATIO")
    match_price_tolerance_bps: float = Field(default=25.0, alias="MATCH_PRICE_TOLERANCE_BPS")
    match_price_tolerance_indication_bps: float = Field(
        default=50.0, alias="MATCH_PRICE_TOLERANCE_INDICATION_BPS"
    )

    # Match dedup windows (plan §10.2)
    match_dedup_window_hours: int = Field(default=24, alias="MATCH_DEDUP_WINDOW_HOURS")
    match_feedback_dedup_window_hours: int = Field(
        default=168, alias="MATCH_FEEDBACK_DEDUP_WINDOW_HOURS"
    )

    # IOI back-stop polling job (plan §13.2, Step 4.6)
    ioi_processing_polling_interval_seconds_market: int = Field(
        default=120, alias="IOI_PROCESSING_POLLING_INTERVAL_SECONDS_MARKET"
    )
    ioi_processing_polling_interval_seconds_off_hours: int = Field(
        default=1800, alias="IOI_PROCESSING_POLLING_INTERVAL_SECONDS_OFF_HOURS"
    )
    ioi_processing_grace_seconds: int = Field(default=30, alias="IOI_PROCESSING_GRACE_SECONDS")
    ioi_processing_max_attempts: int = Field(default=3, alias="IOI_PROCESSING_MAX_ATTEMPTS")

    # Dossier "hot" thresholds (plan §17)
    dossier_hot_trade_hours: int = Field(default=24, alias="DOSSIER_HOT_TRADE_HOURS")
    dossier_hot_inquiry_hours: int = Field(default=4, alias="DOSSIER_HOT_INQUIRY_HOURS")

    # Correlative-attribution job (plan §14.5)
    correlative_attribution_window_hours: int = Field(
        default=24, alias="CORRELATIVE_ATTRIBUTION_WINDOW_HOURS"
    )

    # Reranker: flow polarity penalty weight (plan Step 1.8). Default 0 = disabled.
    rerank_weight_flow_polarity: float = Field(default=0.0, alias="RERANK_WEIGHT_FLOW_POLARITY")

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
