import asyncio
import json
import logging
from unittest.mock import patch
from pprint import pprint
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from csf_recommendation_engine.core.config import get_settings
from csf_recommendation_engine.core.startup import preload_champion_state, preload_client_entity_catalog, preload_intelligence_service
from csf_recommendation_engine.infra.db.pool import init_db_pool, close_db_pool
from csf_recommendation_engine.jobs.rec_refresh_pipeline import run_rec_refresh_pipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")

async def mock_insert_ai_recommendations(conn, rows):
    print("\n" + "="*80)
    print(f"MOCKED INSERT AI RECOMMENDATIONS: {len(rows)} rows generated.")
    for i, row in enumerate(rows, 1):
        print(f"\n[Recommendation {i}]")
        print(f"Entity ID:   {row['entity_id']}")
        print(f"Product:     {row['product']}")
        print(f"Description: {row['description']}")
        print("Details:")
        pprint(json.loads(row['details']), indent=2)
    print("="*80 + "\n")
    return len(rows)

async def mock_insert_cross_block_matches(conn, rows):
    print("\n" + "="*80)
    print(f"MOCKED INSERT CROSS BLOCK MATCHES: {len(rows)} rows generated.")
    for i, row in enumerate(rows, 1):
        print(f"\n[Match {i}]")
        print(f"Product:      {row['product_name']}")
        print(f"Match %:      {row['match_percentage']}%")
        print(f"Buyer:        {row['buyer_entity_id']} ({row['buyer_lots']} lots)")
        print(f"Seller:       {row['seller_entity_id']} ({row['seller_lots']} lots)")
        print(f"Description:  {row['description']}")
    print("="*80 + "\n")
    return len(rows)

async def mock_acquire_advisory_lock(conn):
    return True

async def main():
    settings = get_settings()
    await init_db_pool(str(settings.database_url))
    
    # Needs explicit artifact paths to load properly outside of FastAPI lifespan
    await preload_champion_state(
        model_path=Path(settings.champion_model_path), 
        artifact_path=Path(settings.model_artifacts_path)
    )
    await preload_client_entity_catalog()

    await preload_intelligence_service()
    
    with patch("csf_recommendation_engine.jobs.rec_refresh_pipeline.insert_ai_recommendations", new=mock_insert_ai_recommendations), \
         patch("csf_recommendation_engine.jobs.rec_refresh_pipeline.insert_cross_block_matches", new=mock_insert_cross_block_matches), \
         patch("csf_recommendation_engine.jobs.rec_refresh_pipeline._acquire_advisory_lock", new=mock_acquire_advisory_lock):
        
        print("\nStarting Dry Run of Rec Refresh Pipeline...\n")
        await run_rec_refresh_pipeline()
        print("\nDry Run Complete.\n")
        
    await close_db_pool()

if __name__ == "__main__":
    asyncio.run(main())
