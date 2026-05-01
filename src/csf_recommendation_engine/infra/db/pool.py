import asyncpg
import ssl
from typing import AsyncGenerator
import logging

logger = logging.getLogger(__name__)

db_pool: asyncpg.Pool | None = None


def get_db_pool() -> asyncpg.Pool:
    if db_pool is None:
        raise RuntimeError("Database pool is not initialized.")
    return db_pool

async def init_db_pool(dsn: str) -> None:
    logger.info("Initializing database connection pool...")
    global db_pool
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    db_pool = await asyncpg.create_pool(
        dsn=dsn, min_size=1, max_size=5, ssl=ssl_context
    )
    logger.info("Database connection pool initialized.")

async def close_db_pool() -> None:
    global db_pool
    if db_pool is not None:
        await db_pool.close()

# 2. THE DEPENDENCY: Any endpoint that needs the DB will call this.
async def get_db_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    if db_pool is None:
        raise RuntimeError("Database pool is not initialized.")
        
    # Acquire a connection, yield it to the endpoint, and release it when done.
    async with db_pool.acquire() as connection:
        yield connection


async def get_optional_db_connection() -> AsyncGenerator[asyncpg.Connection | None, None]:
    """Yields a DB connection if configured; otherwise yields None.

    This allows the API to start (and serve /health) even when DATABASE_URL is not set.
    """

    if db_pool is None:
        yield None
        return

    async with db_pool.acquire() as connection:
        yield connection