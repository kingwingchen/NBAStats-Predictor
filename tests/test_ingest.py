"""Integration tests for ingest idempotency.

Run two back-to-back upserts, assert row counts are identical. This guards
against ON CONFLICT bugs that would either duplicate rows or erroneously
drop them on re-run.

These tests hit a live Supabase connection — they require SUPABASE_DB_URL
to be set (locally via .env, in CI via GitHub Actions secret). The entire
module is skipped when the var is absent so unit-only CI environments are
unaffected.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text

from nba_predictor.db.connection import get_engine

pytestmark = pytest.mark.skipif(
    not os.environ.get("SUPABASE_DB_URL"),
    reason="SUPABASE_DB_URL not set — skipping DB integration tests",
)


def _count(table: str) -> int:
    with get_engine().connect() as conn:
        return int(conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one())


def test_players_idempotent():
    from nba_predictor.ingest.players import backfill_players

    n1 = backfill_players()
    n2 = backfill_players()
    assert n1 == n2, f"players: second run returned different count ({n1} → {n2})"
    assert _count("players") == n1


def test_backfill_season_idempotent():
    from nba_predictor.ingest.backfill import backfill_season

    season = "2024-25"
    games1, pgl1 = backfill_season(season)
    games2, pgl2 = backfill_season(season)
    assert games1 == games2, f"games: second run changed count ({games1} → {games2})"
    assert pgl1 == pgl2, f"player_game_logs: second run changed count ({pgl1} → {pgl2})"


def test_team_stats_idempotent():
    from nba_predictor.ingest.team_stats import backfill_team_season

    season = "2024-25"
    n1 = backfill_team_season(season)
    n2 = backfill_team_season(season)
    assert n1 == n2, f"team_game_logs: second run changed count ({n1} → {n2})"
    assert n1 > 0, "team_game_logs: expected at least one row for 2024-25"
