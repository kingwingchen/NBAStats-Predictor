"""Resolve the qualifying player universe for a given prediction date.

A player qualifies if they averaged ≥ MPG_THRESHOLD minutes per game in
the current season OR the prior season. Two-season window rationale:
  - Current season: captures in-form players and role changes mid-year
  - Prior season fallback: handles the first ~10 games of a new season
    before current-season averages are statistically stable (cold-start)
  - Minimum MPG (not games played): a player on a hot streak who just got
    promoted to the rotation qualifies immediately; a veteran missing games
    due to injury doesn't fall out of the universe after a rest
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
from sqlalchemy import text

from nba_predictor.config import MPG_THRESHOLD
from nba_predictor.db.connection import get_engine

logger = logging.getLogger(__name__)

# Derive current and prior season dynamically from the games table so the
# query stays correct when seasons roll over without a code change.
# UNION ALL inside an IN clause — prior_season may be empty (first season),
# which is fine: IN with an empty subquery just excludes that branch.
_UNIVERSE_SQL = """
WITH season_ranks AS (
    SELECT season, RANK() OVER (ORDER BY season DESC) AS rnk
    FROM games
    WHERE game_date <= :as_of
    GROUP BY season
),
current_season AS (SELECT season FROM season_ranks WHERE rnk = 1),
prior_season   AS (SELECT season FROM season_ranks WHERE rnk = 2),
player_mpg AS (
    SELECT
        pgl.player_id,
        g.season,
        AVG(pgl.min) AS avg_min
    FROM player_game_logs pgl
    JOIN games g ON g.game_id = pgl.game_id
    WHERE g.game_date <= :as_of
      AND g.season IN (
          SELECT season FROM current_season
          UNION ALL
          SELECT season FROM prior_season
      )
    GROUP BY pgl.player_id, g.season
),
qualifying AS (
    SELECT DISTINCT player_id
    FROM player_mpg
    WHERE avg_min >= :threshold
)
SELECT p.player_id, p.full_name, p.is_active
FROM players p
JOIN qualifying q ON q.player_id = p.player_id
ORDER BY p.full_name
"""


def get_qualifying_players(as_of_date: date | None = None) -> pd.DataFrame:
    """Return players averaging ≥ MPG_THRESHOLD in current or prior season.

    Parameters
    ----------
    as_of_date:
        Treat all games up to and including this date as known. Defaults to
        today. Must be set explicitly in backtesting contexts to prevent
        leakage (don't pass future data into the universe resolver).

    Returns
    -------
    DataFrame with columns: player_id (int), full_name (str), is_active (bool)
    """
    if as_of_date is None:
        as_of_date = date.today()

    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text(_UNIVERSE_SQL),
            conn,
            params={"as_of": as_of_date, "threshold": MPG_THRESHOLD},
        )

    logger.info(
        "Universe as of %s: %d qualifying players (MPG ≥ %.0f)",
        as_of_date,
        len(df),
        MPG_THRESHOLD,
    )
    return df


if __name__ == "__main__":
    # `uv run python -m nba_predictor.features.player_universe`
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    universe = get_qualifying_players()
    print(universe.head(20).to_string(index=False))
    active = universe["is_active"].sum()
    print(
        f"\nTotal qualifying: {len(universe)}  |  Active: {active}  |  Inactive: {len(universe) - active}"
    )
