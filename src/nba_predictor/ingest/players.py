"""Backfill the `players` table from `CommonAllPlayers`.

This is the smallest ingest task by row count (~3k rows after filtering),
which makes it the right place to validate the **idempotent upsert pattern**
before scaling it up to 100k+ rows of player_game_logs.

Filter rationale:
  We keep only players whose careers overlap with the seasons we'll backfill
  (2020-21 onward). Storing every retired player in NBA history would bloat
  the table without serving a purpose — every FK from player_game_logs is
  guaranteed to land within this filtered universe.

Position is intentionally left NULL in v1: CommonAllPlayers doesn't expose
it, and the only alternative (per-player CommonPlayerInfo calls) would mean
~3000 rate-limited HTTP calls. We can backfill positions in v2 if a feature
needs them.
"""

from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import text

from nba_predictor.config import SEASONS
from nba_predictor.db.connection import get_engine
from nba_predictor.ingest.nba_client import get_client

logger = logging.getLogger(__name__)


# Earliest season we ingest = anchor for "still relevant" filter.
# SEASONS[0] is "2020-21" → the integer 2020.
_EARLIEST_YEAR = int(SEASONS[0].split("-")[0])
_LATEST_YEAR = int(SEASONS[-1].split("-")[0])


_UPSERT_SQL = text(
    """
    INSERT INTO players (player_id, full_name, is_active)
    VALUES (:player_id, :full_name, :is_active)
    ON CONFLICT (player_id) DO UPDATE SET
        full_name = EXCLUDED.full_name,
        is_active = EXCLUDED.is_active
    """
)


def _select_relevant(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep players whose career overlaps with our ingest window.

    `FROM_YEAR` / `TO_YEAR` are the player's first / last NBA season as
    integers (e.g. 2020). A career overlaps [2020-21, 2025-26] iff
    FROM_YEAR <= 2025 and TO_YEAR >= 2020.
    """
    # nba_api ships these as strings sometimes ("2020") and ints other times;
    # coerce so the comparison is unambiguous.
    from_year = pd.to_numeric(raw["FROM_YEAR"], errors="coerce")
    to_year = pd.to_numeric(raw["TO_YEAR"], errors="coerce")
    mask = (from_year <= _LATEST_YEAR) & (to_year >= _EARLIEST_YEAR)
    return raw.loc[mask].copy()


def _to_rows(df: pd.DataFrame) -> list[dict]:
    """Map the nba_api column shape to our `players` table columns."""
    return [
        {
            "player_id": int(row.PERSON_ID),
            "full_name": str(row.DISPLAY_FIRST_LAST),
            "is_active": bool(int(row.ROSTERSTATUS) == 1),
        }
        for row in df.itertuples(index=False)
    ]


def backfill_players() -> int:
    """Pull CommonAllPlayers, filter, upsert. Returns rows touched."""
    client = get_client()
    raw = client.get_all_players(only_current_season=False)
    logger.info("Fetched %d player records (all-time)", len(raw))

    relevant = _select_relevant(raw)
    rows = _to_rows(relevant)
    logger.info(
        "Filtered to %d players overlapping seasons %s–%s",
        len(rows),
        SEASONS[0],
        SEASONS[-1],
    )

    engine = get_engine()
    with engine.begin() as conn:
        # SQLAlchemy turns a list[dict] payload into a single executemany
        # call → one round-trip, atomic via the surrounding transaction.
        conn.execute(_UPSERT_SQL, rows)

    logger.info("Upserted %d player rows", len(rows))
    return len(rows)


def count_players() -> int:
    engine = get_engine()
    with engine.connect() as conn:
        return int(conn.execute(text("SELECT COUNT(*) FROM players")).scalar_one())


if __name__ == "__main__":
    # `uv run python -m nba_predictor.ingest.players`
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    backfill_players()
    logger.info("players table now contains %d rows", count_players())
