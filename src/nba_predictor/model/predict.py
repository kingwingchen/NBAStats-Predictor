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
from nba_predictor.features.build import STAT_X_COLS

logger = logging.getLogger(__name__)


def load_model(
    stat: str = "pts",
    artifact_path: str | Path | None = None,
) -> tuple[XGBRegressor, int | None]:
    """Load a trained XGBoost model.

    Parameters
    ----------
    stat:
        Which per-stat model to load ('pts', 'reb', 'ast', 'fg3m'). The latest
        model_runs row for that stat is used. Defaults to 'pts' so V1 callers
        that pass no arguments continue to work unchanged.
    artifact_path:
        Path to a .json model file. When provided, stat is ignored and
        run_id is returned as None.

    Returns
    -------
    (model, run_id) — run_id is None when artifact_path is provided directly.
    """
    if artifact_path is not None:
        return _load_from_path(Path(artifact_path)), None

    # Look up latest model_run for this stat
    with get_engine().connect() as conn:
        row = conn.execute(
            text(
                "SELECT run_id, artifact_path FROM model_runs "
                "WHERE stat = :stat ORDER BY trained_at DESC LIMIT 1"
            ),
            {"stat": stat},
        ).fetchone()

    if row is None:
        raise RuntimeError(
            f"No model_runs rows for stat='{stat}' — "
            f"run `uv run python -m nba_predictor.model.train --stat {stat}` first."
        )

    run_id, path_str = row
    logger.info("Loading model [%s] from run_id=%d: %s", stat, run_id, path_str)
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


def predict(
    features_df: pd.DataFrame,
    stat: str = "pts",
    model: XGBRegressor | None = None,
) -> pd.DataFrame:
    """Score a feature DataFrame and return predictions for one stat.

    Parameters
    ----------
    features_df:
        Output of build_features(). Rows with NaN in feature columns are kept
        but will produce NaN predictions — callers decide whether to drop.
    stat:
        Target stat ('pts', 'reb', 'ast', 'fg3m'). Selects STAT_X_COLS[stat]
        as inputs and names the output column predicted_{stat}.
    model:
        Pre-loaded XGBRegressor. When None, loads the latest model for stat from DB.

    Returns
    -------
    DataFrame with columns: player_id, game_id, game_date, predicted_{stat}.
    Sorted by predicted_{stat} descending.
    """
    if model is None:
        model, _ = load_model(stat=stat)

    x_cols = STAT_X_COLS[stat]
    pred_col = f"predicted_{stat}"

    # XGBoost handles NaN natively; cold-start rows pass through with degraded predictions.
    preds = model.predict(features_df[x_cols])

    result = features_df[["player_id", "game_id", "game_date"]].copy()
    result[pred_col] = preds
    result = result.sort_values(pred_col, ascending=False).reset_index(drop=True)

    logger.info(
        "Predicted [%s] %d rows. Top: player_id=%s %s=%.1f",
        stat,
        len(result),
        result["player_id"].iloc[0] if len(result) else "n/a",
        pred_col,
        result[pred_col].iloc[0] if len(result) else float("nan"),
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
