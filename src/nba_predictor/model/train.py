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
from nba_predictor.features.build import STAT_X_COLS, build_features

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
    stat: str = "pts",
    fold_days: int = _FOLD_DAYS,
    min_train_days: int = _MIN_TRAIN_DAYS,
    val_start_date: date | None = None,
    trial: optuna.Trial | None = None,
) -> tuple[list[float], np.ndarray]:
    """Expanding-window walk-forward CV.

    Returns
    -------
    (fold_maes, residuals)
        fold_maes  : per-fold MAE values (len = number of valid folds)
        residuals  : all validation residuals concatenated (actual − predicted),
                     used by Phase C to fit a Gaussian for P(Over/Under) calibration.
                     Empty array when val_start_date is set (Optuna fast-path — discard).

    Parameters
    ----------
    df:
        Feature DataFrame from build_features(). Must contain the target stat
        and all columns in STAT_X_COLS[stat]. NaN rows are dropped per fold.
    params:
        XGBRegressor hyperparameters. `n_estimators` is the tree-count ceiling;
        early stopping determines actual usage per fold.
    stat:
        Target stat ('pts', 'reb', 'ast', 'fg3m'). Selects the matching feature
        set from STAT_X_COLS and uses that column as the label.
    val_start_date:
        Only create folds where the validation window starts on or after this
        date. Set to (max_date - 365 days) for fast Optuna tuning; leave None
        for the full CV used in final reporting.
    trial:
        Optuna trial object. When provided, reports fold MAEs for pruning.
    """
    import pandas as pd  # local to avoid circular import

    x_cols = STAT_X_COLS[stat]
    clean = df.dropna(subset=[stat, *x_cols]).copy()
    clean["game_date"] = pd.to_datetime(clean["game_date"])
    clean = clean.sort_values("game_date").reset_index(drop=True)

    dates = clean["game_date"]
    min_date = dates.min().date()
    max_date = dates.max().date()

    first_T = min_date + timedelta(days=min_train_days)
    fold_maes: list[float] = []
    all_residuals: list[float] = []
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

        X_tr, y_tr = clean.loc[train_mask, x_cols], clean.loc[train_mask, stat]
        X_val, y_val = clean.loc[val_mask, x_cols], clean.loc[val_mask, stat]

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
        # Residuals = actual − predicted; Gaussian fitted to these in run_training.
        all_residuals.extend((y_val.values - preds).tolist())

        if trial is not None:
            trial.report(fold_mae, step=len(fold_maes) - 1)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        T += timedelta(days=fold_days)

    return fold_maes, np.array(all_residuals)


# ----------------------------------------------------------------------- Optuna


def _make_optuna_objective(df, stat: str = "pts"):
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
        # Residuals ignored here — Optuna uses only the last 12 months of folds (fast-path).
        # Calibration residuals come from the full CV run in run_training.
        fold_maes, _ = walk_forward_cv(df, params, stat=stat, val_start_date=tuning_start, trial=trial)
        if not fold_maes:
            raise optuna.exceptions.TrialPruned()
        return float(np.mean(fold_maes))

    return objective


def tune_hyperparams(df, *, n_trials: int = 50, stat: str = "pts") -> dict[str, Any]:
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
        _make_optuna_objective(df, stat=stat),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best = study.best_params
    logger.info(
        "Optuna finished [%s]: best trial %d, MAE=%.4f, params=%s",
        stat,
        study.best_trial.number,
        study.best_value,
        best,
    )
    return best


# ---------------------------------------------------------------- final model


def train_final_model(df, params: dict[str, Any], stat: str = "pts") -> XGBRegressor:
    """Fit a single XGBoost model on all available training data.

    No validation split here — the walk-forward CV already gave us an
    unbiased MAE estimate. Training on the full dataset maximises the
    signal available for production predictions.
    """
    import pandas as pd  # local import

    x_cols = STAT_X_COLS[stat]
    clean = df.dropna(subset=[stat, *x_cols]).copy()
    clean["game_date"] = pd.to_datetime(clean["game_date"])
    clean = clean.sort_values("game_date")

    model = XGBRegressor(
        **params,
        tree_method="hist",
        verbosity=0,
        random_state=42,
    )
    model.fit(clean[x_cols], clean[stat])
    logger.info("Final model trained on %d rows, %d features (stat=%s)", len(clean), len(x_cols), stat)
    return model


# ------------------------------------------------------------- persistence


def save_and_register(
    model: XGBRegressor,
    params: dict[str, Any],
    cv_mae: float,
    baseline_mae: float,
    train_end_date: date,
    stat: str = "pts",
    residual_mean: float | None = None,
    residual_std: float | None = None,
) -> str:
    """Save model artifact and write a row to model_runs. Returns artifact path."""
    from datetime import datetime

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Stat included in filename so multiple per-stat artifacts coexist in ARTIFACTS_DIR.
    artifact_path = ARTIFACTS_DIR / f"xgb_{stat}_{ts}.json"
    model.save_model(str(artifact_path))
    logger.info("Saved model artifact: %s", artifact_path)

    # Store the JSON content in the DB so CI runners can load it without needing
    # the file on disk — artifact_path is kept as a human-readable archive reference.
    model_json = artifact_path.read_text()

    with get_engine().begin() as conn:
        conn.execute(
            text("""
                INSERT INTO model_runs
                    (trained_at, train_end_date, cv_mae, baseline_mae, params_json,
                     artifact_path, stat, residual_mean, residual_std, model_json)
                VALUES
                    (:trained_at, :train_end_date, :cv_mae, :baseline_mae, :params_json,
                     :artifact_path, :stat, :residual_mean, :residual_std, :model_json)
            """),
            {
                "trained_at": datetime.now(UTC),
                "train_end_date": train_end_date,
                "cv_mae": cv_mae,
                "baseline_mae": baseline_mae,
                "params_json": json.dumps(params),
                "artifact_path": str(artifact_path),
                "stat": stat,
                "residual_mean": residual_mean,
                "residual_std": residual_std,
                "model_json": model_json,
            },
        )
    logger.info(
        "Registered model_run [%s] (cv_mae=%.4f, baseline_mae=%.4f, residual_std=%s)",
        stat, cv_mae, baseline_mae,
        f"{residual_std:.4f}" if residual_std is not None else "None",
    )
    return str(artifact_path)


# ---------------------------------------------------------------- orchestrator


def run_training(as_of_date: date | None = None, *, n_trials: int = 50, stat: str = "pts") -> dict:
    """Full training pipeline for one stat: features → tune → CV → final fit → save.

    Parameters
    ----------
    stat:
        Target stat to train ('pts', 'reb', 'ast', 'fg3m'). Each stat gets
        its own model artifact and model_runs row, enabling independent tuning.

    Returns a summary dict with stat, cv_mae, baseline_mae, artifact_path.
    """
    import pandas as pd  # local import

    from nba_predictor.model.evaluate import compute_baselines

    if as_of_date is None:
        as_of_date = date.today()

    logger.info("=== run_training stat=%s as_of=%s, n_trials=%d ===", stat, as_of_date, n_trials)

    df = build_features(as_of_date=as_of_date)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Baseline MAEs (before any tuning)
    baselines = compute_baselines(df, stat=stat)
    roll10_mae = baselines[f"roll10_{stat}"]
    logger.info(
        "Baselines [%s] — season_avg: %.3f, roll10: %.3f",
        stat,
        baselines[f"season_avg_{stat}"],
        roll10_mae,
    )

    # Hyperparameter search
    logger.info("Starting Optuna search [%s] (%d trials)...", stat, n_trials)
    best_params = tune_hyperparams(df, n_trials=n_trials, stat=stat)

    # Final CV with best params (all folds, for accurate headline metric + residual calibration)
    logger.info("Running full walk-forward CV [%s] with best params...", stat)
    all_fold_maes, all_residuals = walk_forward_cv(df, best_params, stat=stat)
    cv_mae = float(np.mean(all_fold_maes))
    logger.info(
        "Walk-forward CV [%s]: %.4f MAE over %d folds (baseline roll10=%.4f)",
        stat,
        cv_mae,
        len(all_fold_maes),
        roll10_mae,
    )

    if cv_mae >= roll10_mae:
        logger.warning(
            "Model MAE [%s] (%.4f) does NOT beat roll10 baseline (%.4f) — "
            "check features before deploying",
            stat,
            cv_mae,
            roll10_mae,
        )

    # Fit Gaussian to all-folds residuals for P(Over/Under) probability calibration.
    # residual = actual − predicted, so actual ~ N(predicted + residual_mean, residual_std²).
    # A near-zero residual_mean means the model is unbiased.
    from scipy.stats import norm as _norm  # noqa: PLC0415 — local to keep module-level imports clean

    residual_mean, residual_std = (
        (_norm.fit(all_residuals)) if len(all_residuals) >= 30 else (0.0, float("nan"))
    )
    logger.info(
        "Residual distribution [%s]: mean=%.4f, std=%.4f (n=%d residuals)",
        stat, residual_mean, residual_std, len(all_residuals),
    )

    # Final model on all data
    model = train_final_model(df, best_params, stat=stat)
    train_end_date = df["game_date"].max().date()

    artifact_path = save_and_register(
        model, best_params, cv_mae, roll10_mae, train_end_date,
        stat=stat,
        residual_mean=float(residual_mean),
        residual_std=float(residual_std),
    )

    return {
        "stat": stat,
        "cv_mae": cv_mae,
        "baseline_mae": roll10_mae,
        "n_folds": len(all_fold_maes),
        "residual_mean": round(float(residual_mean), 4),
        "residual_std": round(float(residual_std), 4),
        "artifact_path": artifact_path,
        "train_end_date": str(train_end_date),
    }


if __name__ == "__main__":
    # `uv run python -m nba_predictor.model.train [--stat pts|reb|ast|fg3m] [--fast]`
    import logging as _logging
    import sys

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    fast = "--fast" in sys.argv
    _stat = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--stat"), "pts")
    result = run_training(n_trials=5 if fast else 50, stat=_stat)
    print("\n=== Training complete ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
