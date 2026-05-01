import asyncio
import logging

from csf_recommendation_engine.core.config import get_settings
from csf_recommendation_engine.infra.db.pool import init_db_pool, get_db_pool, close_db_pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

async def main():
    settings = get_settings()
    logging.info("Initializing database connection pool for cleanup...")
    await init_db_pool(str(settings.database_url))
    
    pool = get_db_pool()
    async with pool.acquire() as conn:
        query = """
            DELETE FROM public.ai_recommendations 
            WHERE created_by_ai_agent = 'Recommendation Engine'
        """
        logging.info(f"Executing query: {query.strip()}")
        
        # asyncpg conn.execute returns the command tag like "DELETE 15"
        status = await conn.execute(query)
        logging.info(f"Cleanup complete. Database result: {status}")
        
    await close_db_pool()
    logging.info("Database connection closed.")

if __name__ == "__main__":
    asyncio.run(main())
