"""Feature engineering layer — public entry point: `build_features(as_of_date)`.

Returns one row per (player, game) for all games up to `as_of_date`, with
every feature derived from data available BEFORE that game:

  Leak prevention: all stat columns are shift(1)-lagged within player groups
  sorted by game_date before any rolling or expanding operation. Game N's
  features therefore use only games 1..N-1. XGBoost handles NaN natively, so
  cold-start rows are kept and flagged via `is_cold_start` rather than dropped.

Two lag variants are used:
  * Cross-season lag  (groupby player_id)         — for roll5/roll10 windows,
    which should span season boundaries so a player's form at game 1 of a new
    season reflects their end-of-last-season shape.
  * Within-season lag (groupby player_id + season) — for season_avg_pts,
    season_avg_min, and games_played_season, which must reset to NaN at each
    new season start.

Opponent defensive quality (opp_def_rating_roll10, opp_pace_roll10):
  v1 uses season-level stats from team_game_logs (one value per team per
  season), so "roll10" resolves to the opponent's full-season rating. This is
  still a meaningful signal for model quality. v2 upgrade: swap to per-game
  BoxScoreAdvancedV2 to enable true rolling windows.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
from sqlalchemy import text

from nba_predictor.config import COLD_START_THRESHOLD, ROLLING_WINDOWS
from nba_predictor.db.connection import get_engine

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- SQL

_PGL_QUERY = """
SELECT
    pgl.player_id,
    pgl.game_id,
    pgl.team_id,
    pgl.pts,
    pgl.min,
    pgl.fga,
    pgl.reb,
    pgl.ast,
    pgl.fg3m,
    g.game_date,
    g.season,
    g.home_team_id,
    g.away_team_id
FROM player_game_logs pgl
JOIN games g ON g.game_id = pgl.game_id
WHERE g.game_date <= :as_of
ORDER BY pgl.player_id, g.game_date
"""

_TGL_QUERY = """
SELECT tgl.team_id, tgl.game_id, tgl.def_rating, tgl.pace
FROM team_game_logs tgl
JOIN games g ON g.game_id = tgl.game_id
WHERE g.game_date <= :as_of
"""

# Ordered list of columns returned by build_features — used as the final
# selector so callers always get a predictable, documented shape.
FEATURE_COLS = [
    "player_id", "game_id", "game_date", "season",
    # Rolling windows (cross-season, shift(1)) — pts/min/fga from v1; reb/ast/fg3m added v2
    "roll5_pts",  "roll10_pts",
    "roll5_min",  "roll10_min",
    "roll5_fga",  "roll10_fga",
    "roll5_reb",  "roll10_reb",
    "roll5_ast",  "roll10_ast",
    "roll5_fg3m", "roll10_fg3m",
    # Season-scoped expanding stats (within-season shift(1), reset each season)
    "season_avg_pts", "season_avg_min",
    "season_avg_reb", "season_avg_ast", "season_avg_fg3m",
    "games_played_season",
    # Context features
    "rest_days", "is_back_to_back",
    "is_home",
    "opp_def_rating_roll10", "opp_pace_roll10",
    "is_cold_start",
    # Stat targets (NaN on inference rows before the game is played)
    "pts", "reb", "ast", "fg3m",
]  # fmt: skip

# Shared context columns referenced by every per-stat feature set.
_CONTEXT_COLS: list[str] = [
    "rest_days", "is_back_to_back", "is_home",
    "opp_def_rating_roll10", "opp_pace_roll10",
    "is_cold_start", "games_played_season",
]

# Per-stat feature sets for Phase B per-stat models.
# Each model only sees the rolling/season features most predictive for its
# target, plus the shared context columns. Keeping feature sets tight
# reduces overfitting and makes SHAP explanations cleaner in interviews.
STAT_X_COLS: dict[str, list[str]] = {
    "pts": [
        "roll5_pts", "roll10_pts",
        "roll5_min", "roll10_min",
        "roll5_fga", "roll10_fga",
        "season_avg_pts", "season_avg_min",
        *_CONTEXT_COLS,
    ],
    "reb": [
        "roll5_reb", "roll10_reb",
        "roll5_min", "roll10_min",
        "season_avg_reb", "season_avg_min",
        *_CONTEXT_COLS,
    ],
    "ast": [
        "roll5_ast", "roll10_ast",
        "roll5_min", "roll10_min",
        "season_avg_ast", "season_avg_min",
        *_CONTEXT_COLS,
    ],
    "fg3m": [
        "roll5_fg3m", "roll10_fg3m",
        "roll5_min", "roll10_min",
        "season_avg_fg3m", "season_avg_min",
        *_CONTEXT_COLS,
    ],
}

# Backward-compatible alias — V1 code that imports X_COLS continues to work
# unchanged. Phase B will migrate callers to STAT_X_COLS[stat] directly.
X_COLS: list[str] = STAT_X_COLS["pts"]


# ----------------------------------------------------------------------- loaders


def _load_pgl(conn, as_of: date) -> pd.DataFrame:
    df = pd.read_sql(text(_PGL_QUERY), conn, params={"as_of": as_of})
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _load_tgl(conn, as_of: date) -> pd.DataFrame:
    return pd.read_sql(text(_TGL_QUERY), conn, params={"as_of": as_of})


# ---------------------------------------------------------------- feature builders


def _player_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all per-player rolling, season, and context features."""
    short_w, long_w = ROLLING_WINDOWS  # (5, 10) from config

    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    # --- Cross-season lags (for roll5 / roll10 spanning season boundaries) ---
    # reb/ast/fg3m added in v2; cross-season so a player's form at game 1 of a
    # new season reflects their end-of-last-season shape.
    for col in ("pts", "min", "fga", "reb", "ast", "fg3m"):
        df[f"_x_{col}"] = df.groupby("player_id")[col].shift(1)

    for stat in ("pts", "min", "fga", "reb", "ast", "fg3m"):
        lag = f"_x_{stat}"
        df[f"roll{short_w}_{stat}"] = df.groupby("player_id")[lag].transform(
            lambda s: s.rolling(short_w, min_periods=1).mean()
        )
        df[f"roll{long_w}_{stat}"] = df.groupby("player_id")[lag].transform(
            lambda s: s.rolling(long_w, min_periods=1).mean()
        )

    # --- Within-season lags (season stats must reset at each new season) ---
    # reb/ast/fg3m added in v2 to support per-stat season averages.
    for col in ("pts", "min", "reb", "ast", "fg3m"):
        df[f"_s_{col}"] = df.groupby(["player_id", "season"])[col].shift(1)

    for stat in ("pts", "min", "reb", "ast", "fg3m"):
        df[f"season_avg_{stat}"] = df.groupby(["player_id", "season"])[f"_s_{stat}"].transform(
            lambda s: s.expanding(min_periods=1).mean()
        )

    # games_played_season = count of non-null season-lag rows = games before current
    df["games_played_season"] = df.groupby(["player_id", "season"])["_s_pts"].transform(
        lambda s: s.notna().cumsum()
    )

    # --- Rest days (capped at 7; NaN for a player's very first game) ---
    prev_date = df.groupby("player_id")["game_date"].shift(1)
    df["rest_days"] = (df["game_date"] - prev_date).dt.days.clip(upper=7)
    df["is_back_to_back"] = (df["rest_days"] == 1).astype(int)

    # --- Home/away ---
    df["is_home"] = (df["team_id"] == df["home_team_id"]).astype(int)

    # --- Cold-start flag ---
    df["is_cold_start"] = (df["games_played_season"] < COLD_START_THRESHOLD).astype(int)

    # Drop intermediate lag columns
    drop = [c for c in df.columns if c.startswith("_x_") or c.startswith("_s_")]
    return df.drop(columns=drop)


def _opponent_features(df: pd.DataFrame, tgl: pd.DataFrame) -> pd.DataFrame:
    """Join opponent season defensive rating and pace onto each player-game row."""
    # Opponent = whichever of home/away is NOT the player's team
    df["opp_team_id"] = df["away_team_id"].where(
        df["team_id"] == df["home_team_id"], df["home_team_id"]
    )
    opp = tgl.rename(
        columns={
            "team_id": "opp_team_id",
            "def_rating": "opp_def_rating_roll10",
            "pace": "opp_pace_roll10",
        }
    )
    return df.merge(
        opp[["opp_team_id", "game_id", "opp_def_rating_roll10", "opp_pace_roll10"]],
        on=["opp_team_id", "game_id"],
        how="left",
    )


# ----------------------------------------------------------------------- public API


def build_features(as_of_date: date | None = None) -> pd.DataFrame:
    """Build the full feature DataFrame for all games up to `as_of_date`.

    Parameters
    ----------
    as_of_date:
        Include all games on or before this date. Defaults to today.
        Always set explicitly in backtesting/walk-forward CV to prevent
        leakage — passing a future date would include data unavailable
        at the time of prediction.

    Returns
    -------
    DataFrame with columns per `FEATURE_COLS`. One row per (player, game).
    Feature columns are computed from games strictly before each row's
    game_date (shift(1) guarantee). `pts`, `reb`, `ast`, `fg3m` are the
    prediction targets; each per-stat model selects its own via STAT_X_COLS.
    """
    if as_of_date is None:
        as_of_date = date.today()

    logger.info("Building features as of %s...", as_of_date)

    with get_engine().connect() as conn:
        df = _load_pgl(conn, as_of_date)
        tgl = _load_tgl(conn, as_of_date)

    logger.info("Loaded %d player-game rows, %d team-game rows", len(df), len(tgl))

    df = _player_features(df)
    df = _opponent_features(df, tgl)

    result = df[FEATURE_COLS].reset_index(drop=True)
    logger.info(
        "Feature DataFrame: %d rows × %d cols, %d unique players",
        len(result),
        len(result.columns),
        result["player_id"].nunique(),
    )
    return result


if __name__ == "__main__":
    # `uv run python -m nba_predictor.features.build`
    import logging as _logging
    from datetime import date as _date

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    df = build_features(as_of_date=_date(2025, 4, 20))
    print(
        df[
            [
                "player_id",
                "game_date",
                "roll10_pts", "roll10_reb", "roll10_ast", "roll10_fg3m",
                "season_avg_pts", "season_avg_reb",
                "opp_def_rating_roll10",
                "pts", "reb", "ast", "fg3m",
            ]
        ]
        .tail(10)
        .to_string(index=False)
    )
    print(f"\nNaN rates:\n{df.isna().mean().round(3).to_string()}")
