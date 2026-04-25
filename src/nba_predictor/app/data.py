"""Streamlit data-access layer: SQL queries with @st.cache_data for the dashboard.

All public functions are decorated with @st.cache_data(ttl=3600). Streamlit
reruns the entire script on every user interaction, so caching every DB call
at the module level is essential — without it, a single selectbox change
would re-execute every query.

Query design: rather than calling build_features() (which loads ~100k rows of
historical data), each query fetches exactly what the dashboard needs. The
CTE in load_tonights_slate() computes roll10/rest_days in SQL using a cheap
indexed scan of recent player_game_logs.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import streamlit as st
from sqlalchemy import text

from nba_predictor.db.connection import get_engine
from nba_predictor.features.player_universe import get_qualifying_players

logger = logging.getLogger(__name__)

# Static mapping: nba_api integer team_id → 3-letter abbreviation.
# These IDs are stable across NBA API versions and never change.
NBA_TEAMS: dict[int, str] = {
    1610612737: "ATL", 1610612738: "BOS", 1610612751: "BKN",
    1610612766: "CHA", 1610612741: "CHI", 1610612739: "CLE",
    1610612742: "DAL", 1610612743: "DEN", 1610612765: "DET",
    1610612744: "GSW", 1610612745: "HOU", 1610612754: "IND",
    1610612746: "LAC", 1610612747: "LAL", 1610612763: "MEM",
    1610612748: "MIA", 1610612749: "MIL", 1610612750: "MIN",
    1610612740: "NOP", 1610612752: "NYK", 1610612760: "OKC",
    1610612753: "ORL", 1610612755: "PHI", 1610612756: "PHX",
    1610612757: "POR", 1610612758: "SAC", 1610612759: "SAS",
    1610612761: "TOR", 1610612762: "UTA", 1610612764: "WAS",
}

# ---------------------------------------------------------------------------
# SQL templates
# ---------------------------------------------------------------------------

_SLATE_SQL = """
WITH pred_date AS (
    SELECT MAX(prediction_date) AS d FROM predictions
),
ranked_pgl AS (
    -- Rank each player's game logs newest-first so we can efficiently
    -- aggregate "last N games" stats without a full table scan.
    SELECT
        pgl.player_id,
        pgl.pts,
        pgl.team_id,
        g.game_date,
        ROW_NUMBER() OVER (
            PARTITION BY pgl.player_id ORDER BY g.game_date DESC
        ) AS rn
    FROM player_game_logs pgl
    JOIN games g ON g.game_id = pgl.game_id
    WHERE g.game_date < (SELECT d FROM pred_date)
),
player_stats AS (
    SELECT
        player_id,
        AVG(pts) FILTER (WHERE rn <= 10)     AS roll10_pts,
        MAX(game_date) FILTER (WHERE rn = 1) AS last_game_date,
        MAX(team_id)   FILTER (WHERE rn = 1) AS latest_team_id
    FROM ranked_pgl
    GROUP BY player_id
)
SELECT
    p.player_id,
    COALESCE(pl.full_name, p.player_id::text)  AS full_name,
    ROUND(p.predicted_pts::numeric, 1)          AS predicted_pts,
    p.prediction_date,
    p.game_id,
    ROUND(ps.roll10_pts::numeric, 1)            AS roll10_pts,
    ps.latest_team_id                           AS team_id,
    g.home_team_id,
    g.away_team_id,
    CASE WHEN ps.latest_team_id = g.home_team_id THEN 1 ELSE 0 END AS is_home,
    CASE
        WHEN ps.latest_team_id = g.home_team_id THEN g.away_team_id
        ELSE g.home_team_id
    END                                          AS opp_team_id,
    (p.prediction_date - ps.last_game_date)::int AS rest_days
FROM predictions p
JOIN games g ON g.game_id = p.game_id
LEFT JOIN players pl ON pl.player_id = p.player_id
LEFT JOIN player_stats ps ON ps.player_id = p.player_id
WHERE p.prediction_date = (SELECT d FROM pred_date)
ORDER BY p.predicted_pts DESC
"""

_MODEL_SQL = """
SELECT run_id, trained_at, train_end_date, cv_mae, baseline_mae
FROM model_runs
ORDER BY trained_at DESC
LIMIT 1
"""

# Load 30 games: 20 to display + 10 warm-up rows so the roll10
# window is fully populated for the very first displayed game.
_HISTORY_SQL = """
WITH ranked AS (
    SELECT
        pgl.pts,
        pgl.min   AS minutes,
        pgl.fga,
        g.game_date,
        g.season,
        ROW_NUMBER() OVER (ORDER BY g.game_date DESC) AS rn
    FROM player_game_logs pgl
    JOIN games g ON g.game_id = pgl.game_id
    WHERE pgl.player_id = :player_id
)
SELECT game_date, pts, minutes, fga, season
FROM ranked
WHERE rn <= 30
ORDER BY game_date ASC
"""

# Season-average opponent defensive rating and pace — used as the
# opp_def_rating_roll10 / opp_pace_roll10 features in the model.
# "roll10" is a slight misnomer in v1: it's the full-season average,
# not a true rolling window. v2 upgrade: per-game BoxScoreAdvancedV2.
_OPP_STATS_SQL = """
SELECT
    AVG(tgl.def_rating) AS def_rating,
    AVG(tgl.pace)       AS pace
FROM team_game_logs tgl
JOIN games g ON g.game_id = tgl.game_id
WHERE tgl.team_id = :team_id
  AND g.season = (
      SELECT season FROM games ORDER BY game_date DESC LIMIT 1
  )
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def load_tonights_slate() -> pd.DataFrame:
    """Return tonight's predictions enriched with team, roll10, and rest info.

    The CTE computes roll10_pts and rest_days in SQL so the dashboard
    doesn't need to load the full feature pipeline. Returns an empty
    DataFrame when no predictions exist (off-season, pipeline not run).
    """
    with get_engine().connect() as conn:
        df = pd.read_sql(text(_SLATE_SQL), conn)

    if df.empty:
        return df

    df["team"] = df["team_id"].map(NBA_TEAMS).fillna("UNK")
    df["opp"] = df["opp_team_id"].map(NBA_TEAMS).fillna("UNK")
    df["home_away"] = df["is_home"].map({1: "Home", 0: "Away"})
    return df


@st.cache_data(ttl=3600)
def load_model_info() -> dict | None:
    """Return the latest model_runs row as a dict, or None if no model exists."""
    with get_engine().connect() as conn:
        row = conn.execute(text(_MODEL_SQL)).fetchone()
    if row is None:
        return None
    return dict(row._mapping)


@st.cache_data(ttl=3600)
def load_qualifying_players() -> pd.DataFrame:
    """Return qualifying players (≥15 MPG) for the dropdown.

    Delegates to the pipeline's player_universe logic so the qualifying
    criteria stay in one place — no duplication of the MPG filter here.
    """
    return get_qualifying_players(as_of_date=date.today())


@st.cache_data(ttl=3600)
def load_player_history(player_id: int) -> pd.DataFrame:
    """Return the last 30 game logs for a player, sorted oldest → newest."""
    with get_engine().connect() as conn:
        df = pd.read_sql(text(_HISTORY_SQL), conn, params={"player_id": player_id})
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


@st.cache_data(ttl=3600)
def load_opp_season_stats(opp_team_id: int) -> dict:
    """Return current-season avg def_rating and pace for the opponent."""
    with get_engine().connect() as conn:
        row = conn.execute(text(_OPP_STATS_SQL), {"team_id": opp_team_id}).fetchone()
    if row is None or row.def_rating is None:
        return {}
    return {
        "opp_def_rating_roll10": round(float(row.def_rating), 1),
        "opp_pace_roll10": round(float(row.pace), 1),
    }


def compute_player_feature_snapshot(
    history_df: pd.DataFrame,
    rest_days: int | None = None,
    is_home: int | None = None,
    opp_stats: dict | None = None,
) -> dict[str, float]:
    """Derive approximate feature values from a player's recent game history.

    Mirrors X_COLS from features/build.py using the same shift(1) convention:
    roll windows use games BEFORE the most recent played game (iloc[:-1]),
    matching what the model actually received at inference time.

    Not cached — called with an already-cached history_df, so the underlying
    DB read is already free. Pure pandas, no I/O.
    """
    if history_df.empty:
        return {}

    opp_stats = opp_stats or {}
    current_season = history_df["season"].iloc[-1]
    season_rows = history_df[history_df["season"] == current_season]
    # Exclude the most recent game to replicate the within-season shift(1)
    season_prev = season_rows.iloc[:-1] if len(season_rows) > 1 else pd.DataFrame()

    # All lags use iloc[:-1] — games before the most recent played game
    pts  = history_df["pts"].iloc[:-1]
    mins = history_df["minutes"].iloc[:-1]
    fga  = history_df["fga"].iloc[:-1]

    def _safe_mean(s: pd.Series) -> float:
        return round(float(s.mean()), 1) if not s.empty else float("nan")

    snapshot: dict[str, float] = {
        "roll5_pts":         _safe_mean(pts.tail(5)),
        "roll10_pts":        _safe_mean(pts.tail(10)),
        "roll5_min":         _safe_mean(mins.tail(5)),
        "roll10_min":        _safe_mean(mins.tail(10)),
        "roll5_fga":         _safe_mean(fga.tail(5)),
        "roll10_fga":        _safe_mean(fga.tail(10)),
        "season_avg_pts":    _safe_mean(season_prev["pts"]) if not season_prev.empty else float("nan"),
        "season_avg_min":    _safe_mean(season_prev["minutes"]) if not season_prev.empty else float("nan"),
        "games_played_season": float(len(season_prev)),
        "rest_days":         float(rest_days) if rest_days is not None else float("nan"),
        "is_back_to_back":   float(1 if rest_days == 1 else 0) if rest_days is not None else float("nan"),
        "is_home":           float(is_home) if is_home is not None else float("nan"),
        "is_cold_start":     float(1 if len(season_prev) < 10 else 0),
        **{k: float(v) for k, v in opp_stats.items()},
    }
    return snapshot
