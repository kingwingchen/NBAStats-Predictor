"""Daily inference pipeline — run once per day after overnight game results land.

Execution order (mirrors the GitHub Actions daily_ingest.yml workflow):
  1. ingest_game_date(yesterday)     — pull last night's results into DB
  2. backfill_actual_pts(yesterday)  — fill in actuals for yesterday's preds
  3. get_tonight_games(today)        — tonight's scheduled games from ScoreboardV2
  4. build_inference_features(...)   — latest player form + tonight's context
  5. predict(features_df)            — score with the active model
  6. write_predictions(...)          — upsert into predictions table

M4 milestone: predictions table has one row per qualifying player on tonight's
slate with predicted_pts populated and actual_pts = NULL (filled tomorrow).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import psycopg2.extras
from sqlalchemy import text

from nba_predictor.db.connection import get_engine
from nba_predictor.features.build import X_COLS, build_features
from nba_predictor.features.player_universe import get_qualifying_players
from nba_predictor.ingest.daily import backfill_actual_pts, ingest_game_date
from nba_predictor.ingest.nba_client import get_client
from nba_predictor.model.predict import load_model, predict

logger = logging.getLogger(__name__)

_PRED_UPSERT_SQL = """
INSERT INTO predictions (model_run_id, player_id, game_id, prediction_date, predicted_pts)
VALUES %s
ON CONFLICT (model_run_id, player_id, game_id) DO UPDATE SET
    predicted_pts = EXCLUDED.predicted_pts,
    prediction_date = EXCLUDED.prediction_date
"""


# ----------------------------------------------------------------- helpers


def get_tonight_games(game_date: date) -> list[dict]:
    """Return tonight's scheduled games as a list of dicts.

    Each dict: {game_id, home_team_id, away_team_id, game_date}.
    Returns [] on off-season days or when ScoreboardV2 is unavailable.
    """
    try:
        header = get_client().get_scoreboard(game_date)
    except Exception as exc:
        logger.warning("ScoreboardV2 unavailable for %s: %s", game_date, exc)
        return []

    if header.empty:
        logger.info("No games scheduled for %s", game_date)
        return []

    games = [
        {
            "game_id": str(row.GAME_ID),
            "home_team_id": int(row.HOME_TEAM_ID),
            "away_team_id": int(row.VISITOR_TEAM_ID),
            "game_date": game_date,
        }
        for row in header.itertuples(index=False)
    ]
    logger.info("%d games scheduled for %s", len(games), game_date)
    return games


def _get_player_current_teams(player_ids: set[int]) -> dict[int, int]:
    """Return {player_id: most_recent_team_id} from player_game_logs."""
    with get_engine().connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT DISTINCT ON (pgl.player_id) pgl.player_id, pgl.team_id
                FROM player_game_logs pgl
                JOIN games g ON g.game_id = pgl.game_id
                WHERE pgl.player_id = ANY(:pids)
                ORDER BY pgl.player_id, g.game_date DESC
            """),
            conn,
            params={"pids": list(player_ids)},
        )
    return dict(zip(df["player_id"].astype(int), df["team_id"].astype(int), strict=True))


def _get_all_team_latest_stats() -> dict[int, dict]:
    """Return {team_id: {def_rating, pace}} — most recent season stats per team."""
    with get_engine().connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT DISTINCT ON (tgl.team_id) tgl.team_id, tgl.def_rating, tgl.pace
                FROM team_game_logs tgl
                JOIN games g ON g.game_id = tgl.game_id
                ORDER BY tgl.team_id, g.game_date DESC
            """),
            conn,
        )
    return {
        int(r.team_id): {"def_rating": r.def_rating, "pace": r.pace}
        for r in df.itertuples(index=False)
    }


def build_inference_features(
    qualifying_ids: set[int],
    tonight_games: list[dict],
) -> pd.DataFrame:
    """Build one feature row per qualifying player scheduled to play tonight.

    Strategy: take each player's most recent historical game row (their current
    form), then overwrite the context features that depend on tonight's matchup:
      - rest_days / is_back_to_back (today - last_game_date)
      - is_home            (player's team == tonight's home_team_id)
      - opp_def_rating_roll10 / opp_pace_roll10  (tonight's opponent)

    Rolling and season features (roll5_pts, season_avg_pts, etc.) are left as-is
    from the historical row — they represent the player's form going into tonight.
    """
    yesterday = date.today() - timedelta(days=1)
    hist_df = build_features(as_of_date=yesterday)
    hist_df["game_date"] = pd.to_datetime(hist_df["game_date"])

    # Latest played game row per qualifying player = their current form
    latest = (
        hist_df[hist_df["player_id"].isin(qualifying_ids)]
        .sort_values("game_date")
        .groupby("player_id")
        .tail(1)
        .reset_index(drop=True)
        .copy()
    )

    if latest.empty:
        logger.warning(
            "No historical rows for qualifying players — cannot build inference features"
        )
        return pd.DataFrame()

    # Map each team to its game info for tonight
    team_to_game: dict[int, dict] = {}
    for game in tonight_games:
        hid, vid = game["home_team_id"], game["away_team_id"]
        team_to_game[hid] = {"game_id": game["game_id"], "is_home": 1, "opp_team_id": vid}
        team_to_game[vid] = {"game_id": game["game_id"], "is_home": 0, "opp_team_id": hid}

    player_teams = _get_player_current_teams(qualifying_ids)
    team_stats = _get_all_team_latest_stats()
    today = date.today()

    rows = []
    for _, row in latest.iterrows():
        pid = int(row["player_id"])
        team_id = player_teams.get(pid)
        if team_id is None or team_id not in team_to_game:
            continue  # player not in tonight's slate

        game_info = team_to_game[team_id]
        opp_id = game_info["opp_team_id"]
        opp = team_stats.get(opp_id, {})

        last_date = row["game_date"].date()
        rest = min((today - last_date).days, 7)

        row = row.copy()
        row["game_id"] = game_info["game_id"]
        row["game_date"] = pd.Timestamp(today)
        row["rest_days"] = float(rest)
        row["is_back_to_back"] = int(rest == 1)
        row["is_home"] = game_info["is_home"]
        row["opp_def_rating_roll10"] = opp.get("def_rating")
        row["opp_pace_roll10"] = opp.get("pace")
        rows.append(row)

    if not rows:
        logger.info("No qualifying players matched tonight's teams")
        return pd.DataFrame()

    result = pd.DataFrame(rows)[["player_id", "game_id", "game_date", *X_COLS]].reset_index(
        drop=True
    )
    logger.info("Built inference features for %d players", len(result))
    return result


def write_predictions(
    preds_df: pd.DataFrame,
    model_run_id: int,
    prediction_date: date,
) -> int:
    """Upsert prediction rows into the predictions table. Returns rows written."""
    rows = [
        (
            model_run_id,
            int(r.player_id),
            str(r.game_id),
            prediction_date,
            float(r.predicted_pts),
        )
        for r in preds_df.itertuples(index=False)
    ]
    if not rows:
        return 0

    with get_engine().begin() as conn:
        raw = conn.connection.driver_connection
        with raw.cursor() as cur:
            psycopg2.extras.execute_values(cur, _PRED_UPSERT_SQL, rows, page_size=500)

    logger.info("Wrote %d predictions for %s", len(rows), prediction_date)
    return len(rows)


# ----------------------------------------------------------------- orchestrator


def run_daily(today: date | None = None) -> dict:
    """Run the full daily pipeline. Returns a summary dict.

    Safe to call multiple times for the same date — all DB writes are
    idempotent (ON CONFLICT DO UPDATE / UPDATE WHERE actual_pts IS NULL).
    """
    if today is None:
        today = date.today()
    yesterday = today - timedelta(days=1)

    logger.info("=== run_daily %s ===", today)

    # 1. Ingest yesterday's completed games
    games_n, pgl_n = ingest_game_date(yesterday)

    # 2. Backfill actual_pts into yesterday's predictions
    actuals_n = backfill_actual_pts(yesterday)

    # 3. Tonight's slate
    tonight_games = get_tonight_games(today)
    if not tonight_games:
        logger.info("No games tonight — pipeline complete")
        return {
            "status": "no_games_tonight",
            "date": str(today),
            "games_ingested": games_n,
            "pgl_ingested": pgl_n,
            "actuals_backfilled": actuals_n,
            "predictions_written": 0,
        }

    # 4. Qualifying universe
    universe_df = get_qualifying_players(as_of_date=today)
    qualifying_ids = set(universe_df["player_id"].astype(int))

    # 5. Inference features
    features_df = build_inference_features(qualifying_ids, tonight_games)
    if features_df.empty:
        logger.warning("No inference features — skipping prediction step")
        return {
            "status": "no_features",
            "date": str(today),
            "games_ingested": games_n,
            "pgl_ingested": pgl_n,
            "actuals_backfilled": actuals_n,
            "predictions_written": 0,
        }

    # 6. Load model and predict
    model, run_id = load_model()
    if run_id is None:
        raise RuntimeError("load_model() returned no run_id — check model_runs table")
    preds_df = predict(features_df, model=model)

    # 7. Write to predictions table
    preds_n = write_predictions(preds_df, run_id, today)

    summary = {
        "status": "ok",
        "date": str(today),
        "games_ingested": games_n,
        "pgl_ingested": pgl_n,
        "actuals_backfilled": actuals_n,
        "predictions_written": preds_n,
    }
    logger.info("run_daily complete: %s", summary)
    return summary


if __name__ == "__main__":
    # `uv run python -m nba_predictor.pipelines.run_daily [YYYY-MM-DD]`
    import logging as _logging
    import sys

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(name)s | %(message)s")

    target = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    result = run_daily(today=target)
    print("\n=== Daily pipeline complete ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
