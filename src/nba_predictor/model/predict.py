"""Model inference: load artifact from DB registry and score a feature DataFrame.

The model_runs table is the authoritative registry — the latest row by
trained_at is always the active model. This means retraining deploys
automatically: insert a new model_runs row and the next inference run
picks it up with no code change.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from xgboost import XGBRegressor

from nba_predictor.db.connection import get_engine
from nba_predictor.features.build import X_COLS

logger = logging.getLogger(__name__)


def load_model(artifact_path: str | Path | None = None) -> tuple[XGBRegressor, int | None]:
    """Load a trained XGBoost model.

    Parameters
    ----------
    artifact_path:
        Path to an .json model file. When None, loads the artifact from
        the most recently trained model_run in the DB.

    Returns
    -------
    (model, run_id) — run_id is None when artifact_path is provided directly.
    """
    if artifact_path is not None:
        return _load_from_path(Path(artifact_path)), None

    # Look up latest model_run
    with get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT run_id, artifact_path FROM model_runs ORDER BY trained_at DESC LIMIT 1")
        ).fetchone()

    if row is None:
        raise RuntimeError(
            "No model_runs rows found in DB — run `uv run python -m nba_predictor.model.train` first."
        )

    run_id, path_str = row
    logger.info("Loading model from run_id=%d: %s", run_id, path_str)
    return _load_from_path(Path(path_str)), run_id


def _load_from_path(path: Path) -> XGBRegressor:
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. "
            "Re-run training or check artifact_path in model_runs."
        )
    model = XGBRegressor()
    model.load_model(str(path))
    return model


def predict(features_df: pd.DataFrame, model: XGBRegressor | None = None) -> pd.DataFrame:
    """Score a feature DataFrame and return predictions.

    Parameters
    ----------
    features_df:
        Output of build_features(). Rows with NaN in any X_COL are kept
        but will produce NaN predictions — callers decide whether to drop.
    model:
        Pre-loaded XGBRegressor. When None, loads the latest from DB.

    Returns
    -------
    DataFrame with columns: player_id, game_id, game_date, predicted_pts.
    Sorted by predicted_pts descending.
    """
    if model is None:
        model, _ = load_model()

    # XGBoost predicts NaN for rows where all features are NaN;
    # we let it through so callers can filter cold-start rows themselves.
    preds = model.predict(features_df[X_COLS])

    result = features_df[["player_id", "game_id", "game_date"]].copy()
    result["predicted_pts"] = preds
    result = result.sort_values("predicted_pts", ascending=False).reset_index(drop=True)

    logger.info(
        "Predicted %d rows. Top: player_id=%s predicted_pts=%.1f",
        len(result),
        result["player_id"].iloc[0] if len(result) else "n/a",
        result["predicted_pts"].iloc[0] if len(result) else float("nan"),
    )
    return result


if __name__ == "__main__":
    # `uv run python -m nba_predictor.model.predict`
    import logging as _logging
    from datetime import date

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    from nba_predictor.features.build import build_features
    from nba_predictor.features.player_universe import get_qualifying_players

    universe = get_qualifying_players()
    qualifying_ids = set(universe["player_id"])

    df = build_features(as_of_date=date.today())
    # Filter to qualifying players and their most recent game row
    df = df[df["player_id"].isin(qualifying_ids)]
    latest = df.sort_values("game_date").groupby("player_id").tail(1)

    preds = predict(latest)
    print(preds.head(20).to_string(index=False))
