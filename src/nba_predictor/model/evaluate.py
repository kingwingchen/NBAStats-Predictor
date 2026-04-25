"""Model evaluation: MAE computation and baseline comparison.

Two baselines must be beaten for the model to be considered useful:
  1. season_avg_pts — naive persistence: predict each player's running
     season average. Simple and hard to beat early in the season.
  2. roll10_pts    — short-memory persistence: predict the player's
     10-game rolling average. Captures recent form; strong baseline.

If XGBoost doesn't beat both, the feature set is broken — diagnose
before tuning rather than tuning your way out of a bad feature.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)


def compute_baselines(df: pd.DataFrame) -> dict[str, float]:
    """Compute MAE for naive baselines on rows that have both target and feature.

    Baselines use the pre-computed feature columns (already shift(1)-lagged)
    as predictions, so they are directly comparable to the model's output —
    no separate leakage concern.
    """
    results = {}
    target = df["pts"].dropna()

    for col in ("season_avg_pts", "roll10_pts"):
        shared = df[["pts", col]].dropna()
        if len(shared) < 100:
            logger.warning("%s: fewer than 100 non-NaN rows — baseline may be unreliable", col)
        results[col] = float(mean_absolute_error(shared["pts"], shared[col]))

    logger.info(
        "Baselines — season_avg_pts: %.3f MAE, roll10_pts: %.3f MAE (n=%d rows with target)",
        results.get("season_avg_pts", float("nan")),
        results.get("roll10_pts", float("nan")),
        len(target),
    )
    return results


def evaluate_model(model, df: pd.DataFrame) -> float:
    """Compute overall MAE of a fitted model on df.

    Drops rows where any X_COL or pts is NaN so the comparison is
    apples-to-apples with the baseline computation.
    """
    from nba_predictor.features.build import X_COLS

    clean = df.dropna(subset=["pts", *X_COLS])
    preds = model.predict(clean[X_COLS])
    mae = float(mean_absolute_error(clean["pts"], preds))
    logger.info("Model overall MAE: %.4f (n=%d rows)", mae, len(clean))
    return mae


def print_report(cv_mae: float, baselines: dict[str, float], n_folds: int) -> None:
    """Print a formatted comparison table to stdout."""
    season_avg_mae = baselines.get("season_avg_pts", float("nan"))
    roll10_mae = baselines.get("roll10_pts", float("nan"))

    beat_season = cv_mae < season_avg_mae
    beat_roll10 = cv_mae < roll10_mae

    lines = [
        "",
        "─" * 48,
        "  Walk-forward CV MAE Report",
        "─" * 48,
        f"  XGBoost CV MAE  : {cv_mae:>7.3f} pts  ({n_folds} folds)",
        f"  Baseline roll10  : {roll10_mae:>7.3f} pts  {'✓ beaten' if beat_roll10 else '✗ NOT beaten'}",
        f"  Baseline season  : {season_avg_mae:>7.3f} pts  {'✓ beaten' if beat_season else '✗ NOT beaten'}",
        "─" * 48,
    ]

    if beat_roll10 and beat_season:
        improvement = (roll10_mae - cv_mae) / roll10_mae * 100
        lines.append(f"  M3 milestone: PASSED  ({improvement:.1f}% better than roll10)")
    else:
        lines.append("  M3 milestone: NOT YET — diagnose features before tuning")

    lines.append("─" * 48)
    print("\n".join(lines))


if __name__ == "__main__":
    # `uv run python -m nba_predictor.model.evaluate`
    import logging as _logging
    from datetime import date

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    from nba_predictor.features.build import build_features

    df = build_features(as_of_date=date(2025, 4, 20))
    baselines = compute_baselines(df)
    print_report(
        cv_mae=float("nan"),  # placeholder until model is trained
        baselines=baselines,
        n_folds=0,
    )
