"""Backfill `games` and `player_game_logs` from `PlayerGameLogs`.

Strategy:
  * One nba_api call per season (the endpoint pages internally), so 6
    seasons = 6 HTTP calls. Cheap and friendly to NBA.com.
  * From each season's response, **derive** `games` rows (one per unique
    GAME_ID) before inserting `player_game_logs`, so the FK constraint is
    always satisfied even on a fresh DB.
  * Inserts are bulk via psycopg2's `execute_values`, which packs hundreds
    of rows into one VALUES list — orders of magnitude faster than naive
    executemany. Idempotency comes from `ON CONFLICT (...) DO UPDATE`,
    keyed on the composite PKs we defined in schema.sql.

MATCHUP parsing: nba_api encodes home/away in a string like ``"BOS vs. LAL"``
(BOS is home) or ``"BOS @ LAL"`` (BOS is away). Each game appears in the
season log as ~24 player rows split across both teams, so we can pick the
home TEAM_ID from any "vs." row and the away TEAM_ID from any "@" row.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd
import psycopg2.extras
from sqlalchemy import text

from nba_predictor.config import SEASONS
from nba_predictor.db.connection import get_engine
from nba_predictor.ingest.nba_client import get_client

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- SQL

_GAMES_UPSERT_SQL = """
INSERT INTO games (game_id, game_date, season, home_team_id, away_team_id)
VALUES %s
ON CONFLICT (game_id) DO UPDATE SET
    game_date    = EXCLUDED.game_date,
    season       = EXCLUDED.season,
    home_team_id = EXCLUDED.home_team_id,
    away_team_id = EXCLUDED.away_team_id
"""

_PGL_COLUMNS: tuple[str, ...] = (
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


# ----------------------------------------------------------------- helpers


def _nan_to_none(value: Any) -> Any:
    """Convert pandas NaN / NaT to Python None for psycopg2.

    psycopg2 happily accepts None as SQL NULL, but a float NaN gets bound
    as the literal string 'NaN' which most numeric columns will accept and
    silently corrupt. Hand-converting up front is the safe path.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _coerce_int(value: Any) -> int | None:
    v = _nan_to_none(value)
    return None if v is None else int(v)


def _coerce_float(value: Any) -> float | None:
    v = _nan_to_none(value)
    return None if v is None else float(v)


# ----------------------------------------------------------------- derivation


def derive_games(season_df: pd.DataFrame) -> tuple[list[tuple], set[str]]:
    """Resolve home/away team_ids per GAME_ID.

    Returns ``(rows, valid_game_ids)``. A small number of games per season
    (typically 0–3, e.g. All-Star or Rising Stars exhibitions) carry a
    one-sided MATCHUP string in PlayerGameLogs — only "vs." rows or only
    "@" rows. Those are dropped and reported, and we propagate the surviving
    game_id set so player_game_logs can be filtered to match (otherwise the
    FK from player_game_logs.game_id → games.game_id would explode).
    """
    df = season_df[["GAME_ID", "GAME_DATE", "SEASON_YEAR", "TEAM_ID", "MATCHUP"]].copy()
    df["is_home"] = df["MATCHUP"].str.contains(" vs. ", regex=False)

    home = (
        df[df["is_home"]]
        .groupby("GAME_ID", as_index=False)
        .agg(
            game_date=("GAME_DATE", "first"),
            season=("SEASON_YEAR", "first"),
            home_team_id=("TEAM_ID", "first"),
        )
    )
    away = (
        df[~df["is_home"]].groupby("GAME_ID", as_index=False).agg(away_team_id=("TEAM_ID", "first"))
    )
    games = home.merge(away, on="GAME_ID", how="inner")

    if len(games) != df["GAME_ID"].nunique():
        # Known-acceptable cause: neutral-site games (preseason abroad, Cup
        # Finals in Vegas, All-Star, some play-in games) are reported by
        # nba_api with "@" on BOTH sides of MATCHUP — there's no home row.
        # We log the game_id prefix breakdown so any future regression
        # affecting real regular-season games is obvious in the logs.
        all_ids = set(df["GAME_ID"].astype(str).unique())
        kept_ids = {row["GAME_ID"] for _, row in games.iterrows()}
        dropped = all_ids - kept_ids
        prefix_counts: dict[str, int] = {}
        for gid in dropped:
            prefix_counts[gid[:5]] = prefix_counts.get(gid[:5], 0) + 1
        logger.warning(
            "Dropped %d game(s) with one-sided MATCHUP data — prefixes: %s "
            "(00124=preseason, 00224=regular, 00324=play-in, 00424=playoffs, "
            "00624=all-star)",
            len(dropped),
            prefix_counts,
        )

    rows = [
        (
            str(r.GAME_ID),
            pd.to_datetime(r.game_date).date(),
            str(r.season),
            int(r.home_team_id),
            int(r.away_team_id),
        )
        for r in games.itertuples(index=False)
    ]
    valid_ids = {row[0] for row in rows}
    return rows, valid_ids


def derive_player_game_logs(
    season_df: pd.DataFrame, valid_game_ids: set[str] | None = None
) -> list[tuple]:
    """Map nba_api columns → our `player_game_logs` schema, in column order.

    `valid_game_ids` (when provided) restricts output to game_ids that exist
    in `games`. Required to keep the FK constraint satisfied — see
    `derive_games` for the underlying reason.
    """
    if valid_game_ids is not None:
        season_df = season_df[season_df["GAME_ID"].astype(str).isin(valid_game_ids)]
    return [
        (
            int(r.PLAYER_ID),
            str(r.GAME_ID),
            int(r.TEAM_ID),
            _coerce_float(r.MIN),
            _coerce_int(r.PTS),
            _coerce_int(r.REB),
            _coerce_int(r.AST),
            _coerce_int(r.STL),
            _coerce_int(r.BLK),
            _coerce_int(r.TOV),
            _coerce_int(r.FGM),
            _coerce_int(r.FGA),
            _coerce_int(r.FG3M),
            _coerce_int(r.FG3A),
            _coerce_int(r.FTM),
            _coerce_int(r.FTA),
            _coerce_int(r.PLUS_MINUS),
        )
        for r in season_df.itertuples(index=False)
    ]


# ----------------------------------------------------------------- upsert


def _bulk_upsert(sql: str, rows: list[tuple], *, page_size: int = 1000) -> None:
    """Run a `VALUES %s` upsert via psycopg2.execute_values."""
    if not rows:
        return
    engine = get_engine()
    with engine.begin() as conn:
        # Reach down to the raw psycopg2 cursor to use execute_values.
        # `conn.connection` is SQLAlchemy's DBAPI connection wrapper; its
        # `.driver_connection` is the actual psycopg2 connection.
        raw_conn = conn.connection.driver_connection
        with raw_conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, rows, page_size=page_size)


# ----------------------------------------------------------------- per-season


def backfill_season(season: str) -> tuple[int, int]:
    """Ingest one season. Returns (games_rows, pgl_rows)."""
    logger.info("[%s] fetching PlayerGameLogs...", season)
    df = get_client().get_player_game_logs(season)
    logger.info("[%s] fetched %d player-game rows", season, len(df))

    games_rows, valid_game_ids = derive_games(df)
    pgl_rows = derive_player_game_logs(df, valid_game_ids=valid_game_ids)
    logger.info(
        "[%s] derived %d games, %d player_game_logs", season, len(games_rows), len(pgl_rows)
    )

    # Games first to satisfy the FK from player_game_logs.
    _bulk_upsert(_GAMES_UPSERT_SQL, games_rows)
    _bulk_upsert(_PGL_UPSERT_SQL, pgl_rows)
    logger.info("[%s] upserted ✓", season)
    return len(games_rows), len(pgl_rows)


def backfill_all_seasons() -> dict[str, tuple[int, int]]:
    results: dict[str, tuple[int, int]] = {}
    for season in SEASONS:
        results[season] = backfill_season(season)
    return results


def report_counts() -> None:
    """Print row counts so we can eyeball the M1 milestone."""
    engine = get_engine()
    with engine.connect() as conn:
        for tbl in ("games", "player_game_logs"):
            n = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar_one()
            logger.info("  %s: %d rows", tbl, n)
        per_season = conn.execute(
            text("SELECT season, COUNT(*) AS games FROM games GROUP BY season ORDER BY season")
        ).all()
        for season, n in per_season:
            logger.info("  games[%s]: %d", season, n)


if __name__ == "__main__":
    # `uv run python -m nba_predictor.ingest.backfill`
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    backfill_all_seasons()
    report_counts()
