"""SQLAlchemy engine factory for Supabase Postgres.

We wrap engine creation so every caller (ingest, features, pipelines, tests)
uses the same pool configuration and connection string. Supabase's session
pooler closes idle conns aggressively, so we set `pool_pre_ping=True` to
recycle dead handles transparently instead of surfacing OperationalErrors
mid-pipeline.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text

from nba_predictor.config import SUPABASE_DB_URL

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return a process-wide singleton engine.

    lru_cache ensures every caller shares the same connection pool — creating
    multiple engines would exhaust Supabase's free-tier connection limit fast
    during backfill.
    """
    return create_engine(
        SUPABASE_DB_URL,
        pool_pre_ping=True,  # cheap liveness check; avoids stale-conn errors
        pool_size=5,  # Supabase free tier allows ~60; stay well under
        max_overflow=5,
        future=True,
        connect_args={"sslmode": "require"},  # Supabase enforces SSL
    )


def ping() -> str:
    """Round-trip a trivial query and return the Postgres version string.

    Used by the Phase 0 connection test and by CI as a readiness probe
    before any real ingest/inference work.
    """
    engine = get_engine()
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version()")).scalar_one()
    return str(version)


if __name__ == "__main__":
    # `uv run python -m nba_predictor.db.connection` for a quick smoke test.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("Connecting to Supabase...")
    logger.info("OK: %s", ping())
