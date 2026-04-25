"""Backfill `team_game_logs` from season-level `LeagueDashTeamStats`.

v1 approximation: nba_api's season-level endpoint gives one set of
def_rating / off_rating / pace per team per season. We expand this into
one row per (team_id, game_id) by joining to the games table, so the
feature builder can reference a game-keyed table consistently.

Within-season, a rolling average of these values is flat (every game has the
same season stat), which is equivalent to using the season average as a
context feature — a sensible defensive-quality proxy for v1.

v2 upgrade path: swap the source to BoxScoreAdvancedV2 per-game (~8 k HTTP
calls per backfill) to get real per-game ratings and enable meaningful
rolling windows.
"""

from __future__ import annotations

import logging

import pandas as pd
import psycopg2.extras
from sqlalchemy import text

from nba_predictor.config import SEASONS
from nba_predictor.db.connection import get_engine
from nba_predictor.ingest.nba_client import get_client

logger = logging.getLogger(__name__)

_TGL_UPSERT_SQL = """
INSERT INTO team_game_logs (team_id, game_id, def_rating, off_rating, pace)
VALUES %s
ON CONFLICT (team_id, game_id) DO UPDATE SET
    def_rating = EXCLUDED.def_rating,
    off_rating = EXCLUDED.off_rating,
    pace       = EXCLUDED.pace
"""


def _bulk_upsert(rows: list[tuple]) -> None:
    if not rows:
        return
    engine = get_engine()
    with engine.begin() as conn:
        raw_conn = conn.connection.driver_connection
        with raw_conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, _TGL_UPSERT_SQL, rows, page_size=1000)


def backfill_team_season(season: str) -> int:
    """Ingest one season's team stats into team_game_logs. Returns rows upserted."""
    logger.info("[%s] fetching LeagueDashTeamStats...", season)
    stats_df = get_client().get_team_advanced_per_game(season)
    if stats_df.empty:
        logger.warning("[%s] LeagueDashTeamStats returned no rows — skipping", season)
        return 0

    engine = get_engine()
    with engine.connect() as conn:
        games_df = pd.read_sql(
            text("SELECT game_id, home_team_id, away_team_id FROM games WHERE season = :s"),
            conn,
            params={"s": season},
        )

    if games_df.empty:
        logger.warning("[%s] no games in DB — run backfill.py first", season)
        return 0

    # Season stats keyed by team_id
    stats = stats_df[["TEAM_ID", "DEF_RATING", "OFF_RATING", "PACE"]].copy()
    stats.columns = ["team_id", "def_rating", "off_rating", "pace"]
    stats["team_id"] = stats["team_id"].astype(int)

    # Each game has a home team and an away team — expand to (team_id, game_id) pairs
    home = games_df[["game_id", "home_team_id"]].rename(columns={"home_team_id": "team_id"})
    away = games_df[["game_id", "away_team_id"]].rename(columns={"away_team_id": "team_id"})
    pairs = pd.concat([home, away], ignore_index=True)

    merged = pairs.merge(stats, on="team_id", how="inner")
    if merged.empty:
        logger.warning("[%s] no team overlap between games and stats — check season string", season)
        return 0

    rows = [
        (
            int(r.team_id),
            str(r.game_id),
            float(r.def_rating),
            float(r.off_rating),
            float(r.pace),
        )
        for r in merged.itertuples(index=False)
    ]

    _bulk_upsert(rows)
    logger.info("[%s] upserted %d team_game_logs rows", season, len(rows))
    return len(rows)


def backfill_all_team_seasons() -> dict[str, int]:
    results: dict[str, int] = {}
    for season in SEASONS:
        results[season] = backfill_team_season(season)
    return results


if __name__ == "__main__":
    # `uv run python -m nba_predictor.ingest.team_stats`
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    backfill_all_team_seasons()
    engine = get_engine()
    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM team_game_logs")).scalar_one()
        logger.info("team_game_logs: %d total rows", n)
