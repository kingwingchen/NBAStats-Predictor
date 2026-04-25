"""Apply src/nba_predictor/db/schema.sql to Supabase.

One-shot bootstrap for Phase 0. After this runs successfully, the six
tables + indices exist in the target database. Re-running is safe — every
statement is CREATE ... IF NOT EXISTS.

Usage:
    uv run python scripts/apply_schema.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import inspect, text

from nba_predictor.config import REPO_ROOT
from nba_predictor.db.connection import get_engine

logger = logging.getLogger(__name__)

SCHEMA_PATH = REPO_ROOT / "src" / "nba_predictor" / "db" / "schema.sql"
EXPECTED_TABLES = {
    "games",
    "players",
    "player_game_logs",
    "team_game_logs",
    "model_runs",
    "predictions",
}


def apply_schema(schema_path: Path = SCHEMA_PATH) -> None:
    sql = schema_path.read_text()
    engine = get_engine()
    # exec_driver_sql lets us send the whole DDL script in one go; `text()`
    # would try to bind-param every colon it finds in the SQL.
    with engine.begin() as conn:
        conn.exec_driver_sql(sql)
    logger.info("Applied schema from %s", schema_path)


def verify_tables() -> None:
    engine = get_engine()
    inspector = inspect(engine)
    actual = set(inspector.get_table_names(schema="public"))
    missing = EXPECTED_TABLES - actual
    if missing:
        raise RuntimeError(f"Schema apply incomplete — missing tables: {missing}")
    logger.info("All expected tables present: %s", sorted(EXPECTED_TABLES))

    # Quick sanity check — every table starts empty.
    with engine.connect() as conn:
        for tbl in sorted(EXPECTED_TABLES):
            count = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar_one()
            logger.info("  %s: %d rows", tbl, count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    apply_schema()
    verify_tables()
