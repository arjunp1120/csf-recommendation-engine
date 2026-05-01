import asyncio
import logging
from csf_recommendation_engine.core.config import get_settings
from csf_recommendation_engine.infra.db.pool import init_db_pool, get_db_pool, close_db_pool

logging.basicConfig(level=logging.INFO, format="%(message)s")

async def main():
    settings = get_settings()
    await init_db_pool(str(settings.postgres_dsn))
    pool = get_db_pool()
    
    async with pool.acquire() as conn:
        print("Connecting to DB to remove accidental rows...")
        query = "DELETE FROM public.ai_recommendations WHERE created_by_ai_agent = 'Recommendation Engine'"
        result = await conn.execute(query)
        print(f"Cleanup finished successfully. Database responded with: {result}")
        
    await close_db_pool()

if __name__ == "__main__":
    asyncio.run(main())
