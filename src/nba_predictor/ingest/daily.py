"""Daily ingest: pull one game-date's results into games + player_game_logs,
then backfill actual_pts into any matching predictions rows.

Two entry points:
  ingest_game_date(game_date) — fetch + upsert one day's results.
  backfill_actual_pts(game_date) — UPDATE predictions.actual_pts for that day.

The daily pipeline calls both in sequence:
  1. ingest_game_date(yesterday) → games + player_game_logs land in DB
  2. backfill_actual_pts(yesterday) → yesterday's prediction rows get actuals
"""

from __future__ import annotations

import logging
from datetime import date

import psycopg2.extras
from sqlalchemy import text

from nba_predictor.db.connection import get_engine
from nba_predictor.ingest.backfill import derive_games, derive_player_game_logs
from nba_predictor.ingest.nba_client import get_client

logger = logging.getLogger(__name__)

# Reuse the same SQL as backfill — idempotent ON CONFLICT DO UPDATE.
_GAMES_UPSERT_SQL = """
INSERT INTO games (game_id, game_date, season, home_team_id, away_team_id)
VALUES %s
ON CONFLICT (game_id) DO UPDATE SET
    game_date    = EXCLUDED.game_date,
    season       = EXCLUDED.season,
    home_team_id = EXCLUDED.home_team_id,
    away_team_id = EXCLUDED.away_team_id
"""

_PGL_COLUMNS = (
    "player_id", "game_id", "team_id",
    "min", "pts", "reb", "ast", "stl", "blk", "tov",
    "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "plus_minus",
)  # fmt: skip

_PGL_UPDATE_COLS = tuple(c for c in _PGL_COLUMNS if c not in ("player_id", "game_id"))

_PGL_UPSERT_SQL = f"""
INSERT INTO player_game_logs ({", ".join(_PGL_COLUMNS)})
VALUES %s
ON CONFLICT (player_id, game_id) DO UPDATE SET
    {", ".join(f"{c} = EXCLUDED.{c}" for c in _PGL_UPDATE_COLS)}
"""


def _date_to_season(d: date) -> str:
    """Convert a calendar date to NBA season string (e.g. '2024-25').

    NBA seasons run October → June. A date in Oct–Dec belongs to the season
    starting that year; Jan–Sep belongs to the season that started the prior year.
    """
    year = d.year if d.month >= 10 else d.year - 1
    return f"{year}-{str(year + 1)[-2:]}"


def _bulk_upsert(sql: str, rows: list[tuple]) -> None:
    if not rows:
        return
    with get_engine().begin() as conn:
        raw = conn.connection.driver_connection
        with raw.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, rows, page_size=1000)


def ingest_game_date(game_date: date) -> tuple[int, int]:
    """Fetch all player game logs for game_date and upsert into DB.

    Returns (games_count, player_game_logs_count). Returns (0, 0) when
    no games were played on that date (off-season, rest day, API outage).
    """
    season = _date_to_season(game_date)
    logger.info("[%s] fetching player game logs (season=%s)...", game_date, season)

    df = get_client().get_player_game_logs_for_dates(game_date, game_date, season)
    if df.empty:
        logger.info("[%s] no games found — skipping ingest", game_date)
        return 0, 0

    logger.info("[%s] fetched %d player-game rows", game_date, len(df))

    games_rows, valid_ids = derive_games(df)
    pgl_rows = derive_player_game_logs(df, valid_game_ids=valid_ids)

    _bulk_upsert(_GAMES_UPSERT_SQL, games_rows)
    _bulk_upsert(_PGL_UPSERT_SQL, pgl_rows)

    logger.info("[%s] upserted %d games, %d player_game_logs", game_date, len(games_rows), len(pgl_rows))
    return len(games_rows), len(pgl_rows)


def backfill_actual_pts(game_date: date) -> int:
    """UPDATE predictions.actual_pts for predictions whose game was played on game_date.

    Joins predictions → player_game_logs → games on (player_id, game_id) and
    only touches rows where actual_pts IS NULL, making repeated calls safe.
    Returns the number of rows updated.
    """
    with get_engine().begin() as conn:
        result = conn.execute(
            text("""
                UPDATE predictions p
                SET actual_pts = pgl.pts
                FROM player_game_logs pgl
                JOIN games g ON g.game_id = pgl.game_id
                WHERE p.player_id = pgl.player_id
                  AND p.game_id   = pgl.game_id
                  AND g.game_date = :game_date
                  AND p.actual_pts IS NULL
            """),
            {"game_date": game_date},
        )
        n = result.rowcount

    logger.info("[%s] backfilled actual_pts for %d prediction rows", game_date, n)
    return n


if __name__ == "__main__":
    # `uv run python -m nba_predictor.ingest.daily [YYYY-MM-DD]`
    import logging as _logging
    import sys
    from datetime import timedelta

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(name)s | %(message)s")

    if len(sys.argv) > 1:
        target = date.fromisoformat(sys.argv[1])
    else:
        target = date.today() - timedelta(days=1)

    g, p = ingest_game_date(target)
    logger.info("Done: %d games, %d player_game_logs", g, p)
    n = backfill_actual_pts(target)
    logger.info("Backfilled %d actual_pts", n)
