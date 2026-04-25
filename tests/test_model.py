"""Walk-forward CV sanity tests — no DB required.

Generates synthetic player-game data in memory and verifies the core
properties of the CV machinery:

  1. Returns a non-empty MAE list when given sufficient history.
  2. All per-fold MAEs are positive finite numbers.
  3. CV MAE on a constant-output signal (all pts == mu) approaches zero
     as the model has infinite data to learn from — the mean prediction
     converges to mu and MAE approaches 0.

These tests run in < 5 seconds on any machine because they use small
synthetic DataFrames and n_estimators=10. They do not require
SUPABASE_DB_URL and are never skipped.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from nba_predictor.features.build import X_COLS
from nba_predictor.model.train import walk_forward_cv


def _synthetic_df(
    n_players: int = 20,
    games_per_player: int = 60,
    *,
    pts_mean: float = 18.0,
    pts_std: float = 5.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a minimal feature DataFrame for CV testing.

    All feature columns are drawn from a standard normal distribution so
    the model has *something* to learn. pts is drawn from N(pts_mean, pts_std).
    Dates are sequential starting from 2021-10-01.
    """
    rng = np.random.default_rng(seed)
    start = date(2021, 10, 1)

    records = []
    for pid in range(n_players):
        for g in range(games_per_player):
            game_date = start + timedelta(days=g * 2)  # one game every 2 days
            row: dict = {
                "player_id": pid,
                "game_id": f"game_{pid}_{g}",
                "game_date": pd.Timestamp(game_date),
                "season": "2021-22",
                "pts": max(0.0, float(rng.normal(pts_mean, pts_std))),
            }
            for col in X_COLS:
                row[col] = float(rng.normal(0, 1))
            records.append(row)

    return pd.DataFrame(records)


# Minimal XGBoost params — intentionally weak so the test is fast
_FAST_PARAMS = {
    "n_estimators": 10,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}

# Reduced CV window sizes for tests: synthetic data spans ~120 days,
# so we need min_train_days << 365 and fold_days << 30 to get real folds.
_TEST_CV_KWARGS = {"min_train_days": 30, "fold_days": 15}


@pytest.fixture(scope="module")
def synthetic_df():
    return _synthetic_df()


def test_walk_forward_cv_returns_nonempty_maes(synthetic_df):
    """CV must produce at least one fold for typical training data sizes."""
    fold_maes = walk_forward_cv(synthetic_df, _FAST_PARAMS, **_TEST_CV_KWARGS)
    assert len(fold_maes) > 0, (
        "walk_forward_cv returned zero folds — check min_train_days or data size"
    )


def test_walk_forward_cv_maes_are_positive_finite(synthetic_df):
    """Every per-fold MAE must be a positive, finite number.

    NaN or infinite MAE indicates a numerical issue (e.g. all-NaN predictions,
    wrong feature shapes, or a broken XGBoost fit).
    """
    fold_maes = walk_forward_cv(synthetic_df, _FAST_PARAMS, **_TEST_CV_KWARGS)
    for i, mae in enumerate(fold_maes):
        assert mae > 0, f"Fold {i}: MAE is non-positive ({mae})"
        assert math.isfinite(mae), f"Fold {i}: MAE is not finite ({mae})"


def test_walk_forward_cv_val_start_date_filters_folds(synthetic_df):
    """val_start_date must reduce the number of evaluated folds.

    Passing a late val_start_date should yield fewer folds than running
    without the filter — this confirms the Optuna fast-path is working.
    Synthetic data ends ~2022-01-29; late_start cuts off the early folds.
    """
    all_folds = walk_forward_cv(synthetic_df, _FAST_PARAMS, **_TEST_CV_KWARGS)
    late_start = date(2022, 1, 1)  # only the last ~4 weeks of the synthetic data
    filtered_folds = walk_forward_cv(
        synthetic_df, _FAST_PARAMS, val_start_date=late_start, **_TEST_CV_KWARGS
    )
    assert len(filtered_folds) < len(all_folds), (
        f"val_start_date filter had no effect: {len(filtered_folds)} == {len(all_folds)} folds"
    )


def test_walk_forward_cv_on_constant_target():
    """MAE on a constant-pts signal should be close to zero.

    If every player always scores exactly pts_mean, a well-fitted model
    should predict pts_mean for all rows and achieve near-zero MAE.
    This exercises the full code path end-to-end with a trivially learnable signal.
    """
    df = _synthetic_df(pts_std=0.0)  # all pts == 18.0
    fold_maes = walk_forward_cv(df, _FAST_PARAMS, **_TEST_CV_KWARGS)
    assert len(fold_maes) > 0, "No folds generated for constant-target test"
    mean_mae = float(np.mean(fold_maes))
    # XGBoost won't hit exactly zero due to regularisation + finite trees,
    # but it should be well under 1 point on a constant target.
    assert mean_mae < 1.0, (
        f"CV MAE on constant target should be < 1.0 pts, got {mean_mae:.4f}"
    )
