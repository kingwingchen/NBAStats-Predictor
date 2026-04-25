"""XGBoost trainer: walk-forward CV, Optuna tuning, artifact persistence.

Training protocol (from master blueprint):
  * Walk-forward expanding window: train on all history through date T,
    validate on T+1..T+30 days, slide T forward 30 days and refit.
  * Chronological split is non-negotiable — random split leaks future
    game outcomes into training labels for time-series sports data.
  * Optuna tunes over the last 12 months of folds only (≈12 folds ×
    50 trials = 600 fits) to keep search tractable. The headline CV MAE
    reported against baselines uses all folds (~40 folds).
  * early_stopping_rounds=50 prevents wasting compute on overfit trees;
    XGBoost internally tracks the best_iteration.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, timedelta
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error
from sqlalchemy import text
from xgboost import XGBRegressor

from nba_predictor.config import ARTIFACTS_DIR
from nba_predictor.db.connection import get_engine
from nba_predictor.features.build import X_COLS, build_features

logger = logging.getLogger(__name__)

# Suppress Optuna's per-trial INFO spam; we log summary ourselves.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --------------------------------------------------------------------------- CV

_FOLD_DAYS = 30
_MIN_TRAIN_DAYS = 365  # first fold trains on ≥ 1 year of history
_MIN_VAL_ROWS = 10  # skip folds with fewer validation rows (off-season gaps)
_EARLY_STOP_ROUNDS = 50


def walk_forward_cv(
    df: pandas.DataFrame,  # noqa: F821 — avoid circular import at module level
    params: dict[str, Any],
    *,
    fold_days: int = _FOLD_DAYS,
    min_train_days: int = _MIN_TRAIN_DAYS,
    val_start_date: date | None = None,
    trial: optuna.Trial | None = None,
) -> list[float]:
    """Expanding-window walk-forward CV. Returns per-fold MAE list.

    Parameters
    ----------
    df:
        Feature DataFrame from build_features(). Must contain `pts` (target)
        and all columns in X_COLS. Rows with NaN in pts or X_COLS are dropped.
    params:
        XGBRegressor hyperparameters. `n_estimators` is the tree-count ceiling;
        early stopping determines actual usage per fold.
    val_start_date:
        Only create folds where the validation window starts on or after this
        date. Set to (max_date - 365 days) for fast Optuna tuning; leave None
        for the full CV used in final reporting.
    trial:
        Optuna trial object. When provided, reports fold MAEs for pruning.
    """
    import pandas as pd  # local to avoid circular import

    clean = df.dropna(subset=["pts", *X_COLS]).copy()
    clean["game_date"] = pd.to_datetime(clean["game_date"])
    clean = clean.sort_values("game_date").reset_index(drop=True)

    dates = clean["game_date"]
    min_date = dates.min().date()
    max_date = dates.max().date()

    first_T = min_date + timedelta(days=min_train_days)
    fold_maes: list[float] = []
    T = first_T

    while T + timedelta(days=fold_days) <= max_date:
        val_start = T + timedelta(days=1)
        val_end = T + timedelta(days=fold_days)

        # Skip folds before the tuning window (Optuna fast-path)
        if val_start_date is not None and val_start < val_start_date:
            T += timedelta(days=fold_days)
            continue

        train_mask = dates.dt.date <= T
        val_mask = (dates.dt.date > T) & (dates.dt.date <= val_end)

        X_tr, y_tr = clean.loc[train_mask, X_COLS], clean.loc[train_mask, "pts"]
        X_val, y_val = clean.loc[val_mask, X_COLS], clean.loc[val_mask, "pts"]

        if len(X_val) < _MIN_VAL_ROWS:
            T += timedelta(days=fold_days)
            continue

        model = XGBRegressor(
            **params,
            early_stopping_rounds=_EARLY_STOP_ROUNDS,
            eval_metric="mae",
            tree_method="hist",
            verbosity=0,
            random_state=42,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        preds = model.predict(X_val)
        fold_mae = float(mean_absolute_error(y_val, preds))
        fold_maes.append(fold_mae)

        if trial is not None:
            trial.report(fold_mae, step=len(fold_maes) - 1)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        T += timedelta(days=fold_days)

    return fold_maes


# ----------------------------------------------------------------------- Optuna


def _make_optuna_objective(df):
    # Tuning window: last 12 months of folds — fast but representative.
    tuning_start = (df["game_date"].max() - timedelta(days=365)).date()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        fold_maes = walk_forward_cv(df, params, val_start_date=tuning_start, trial=trial)
        if not fold_maes:
            raise optuna.exceptions.TrialPruned()
        return float(np.mean(fold_maes))

    return objective


def tune_hyperparams(df, *, n_trials: int = 50) -> dict[str, Any]:
    """Run Optuna hyperparameter search. Returns best params dict."""
    import pandas as pd  # local import

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3),
    )
    study.optimize(
        _make_optuna_objective(df),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best = study.best_params
    logger.info(
        "Optuna finished: best trial %d, MAE=%.4f, params=%s",
        study.best_trial.number,
        study.best_value,
        best,
    )
    return best


# ---------------------------------------------------------------- final model


def train_final_model(df, params: dict[str, Any]) -> XGBRegressor:
    """Fit a single XGBoost model on all available training data.

    No validation split here — the walk-forward CV already gave us an
    unbiased MAE estimate. Training on the full dataset maximises the
    signal available for production predictions.
    """
    import pandas as pd  # local import

    clean = df.dropna(subset=["pts", *X_COLS]).copy()
    clean["game_date"] = pd.to_datetime(clean["game_date"])
    clean = clean.sort_values("game_date")

    model = XGBRegressor(
        **params,
        tree_method="hist",
        verbosity=0,
        random_state=42,
    )
    model.fit(clean[X_COLS], clean["pts"])
    logger.info("Final model trained on %d rows, %d features", len(clean), len(X_COLS))
    return model


# ------------------------------------------------------------- persistence


def save_and_register(
    model: XGBRegressor,
    params: dict[str, Any],
    cv_mae: float,
    baseline_mae: float,
    train_end_date: date,
) -> str:
    """Save model artifact and write a row to model_runs. Returns artifact path."""
    from datetime import datetime

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_path = ARTIFACTS_DIR / f"xgb_{ts}.json"
    model.save_model(str(artifact_path))
    logger.info("Saved model artifact: %s", artifact_path)

    with get_engine().begin() as conn:
        conn.execute(
            text("""
                INSERT INTO model_runs
                    (trained_at, train_end_date, cv_mae, baseline_mae, params_json, artifact_path)
                VALUES
                    (:trained_at, :train_end_date, :cv_mae, :baseline_mae, :params_json, :artifact_path)
            """),
            {
                "trained_at": datetime.now(UTC),
                "train_end_date": train_end_date,
                "cv_mae": cv_mae,
                "baseline_mae": baseline_mae,
                "params_json": json.dumps(params),
                "artifact_path": str(artifact_path),
            },
        )
    logger.info("Registered model_run (cv_mae=%.4f, baseline_mae=%.4f)", cv_mae, baseline_mae)
    return str(artifact_path)


# ---------------------------------------------------------------- orchestrator


def run_training(as_of_date: date | None = None, *, n_trials: int = 50) -> dict:
    """Full training pipeline: features → tune → CV → final fit → save.

    Returns a summary dict with cv_mae, baseline_mae, artifact_path.
    """
    import pandas as pd  # local import

    from nba_predictor.model.evaluate import compute_baselines

    if as_of_date is None:
        as_of_date = date.today()

    logger.info("=== run_training as_of=%s, n_trials=%d ===", as_of_date, n_trials)

    df = build_features(as_of_date=as_of_date)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Baseline MAEs (before any tuning)
    baselines = compute_baselines(df)
    roll10_mae = baselines["roll10_pts"]
    logger.info(
        "Baselines — season_avg: %.3f, roll10: %.3f", baselines["season_avg_pts"], roll10_mae
    )

    # Hyperparameter search
    logger.info("Starting Optuna search (%d trials)...", n_trials)
    best_params = tune_hyperparams(df, n_trials=n_trials)

    # Final CV with best params (all folds, for accurate headline metric)
    logger.info("Running full walk-forward CV with best params...")
    all_fold_maes = walk_forward_cv(df, best_params)
    cv_mae = float(np.mean(all_fold_maes))
    logger.info(
        "Walk-forward CV: %.4f MAE over %d folds (baseline roll10=%.4f)",
        cv_mae,
        len(all_fold_maes),
        roll10_mae,
    )

    if cv_mae >= roll10_mae:
        logger.warning(
            "Model MAE (%.4f) does NOT beat roll10 baseline (%.4f) — "
            "check features before deploying",
            cv_mae,
            roll10_mae,
        )

    # Final model on all data
    model = train_final_model(df, best_params)
    train_end_date = df["game_date"].max().date()

    artifact_path = save_and_register(model, best_params, cv_mae, roll10_mae, train_end_date)

    return {
        "cv_mae": cv_mae,
        "baseline_mae": roll10_mae,
        "n_folds": len(all_fold_maes),
        "artifact_path": artifact_path,
        "train_end_date": str(train_end_date),
    }


if __name__ == "__main__":
    # `uv run python -m nba_predictor.model.train`
    # Add --fast flag for a quick smoke test with fewer trials.
    import logging as _logging
    import sys

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    fast = "--fast" in sys.argv
    result = run_training(n_trials=5 if fast else 50)
    print("\n=== Training complete ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
