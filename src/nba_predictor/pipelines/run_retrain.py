"""Biweekly retrain pipeline — rebuild the XGBoost model on all available history.

Execution order:
  1. build_features(today)       — full historical feature DataFrame
  2. tune_hyperparams(df)        — Optuna walk-forward search (last 12 months of folds)
  3. walk_forward_cv(df, params) — all-fold CV for headline MAE metric
  4. train_final_model(df)       — fit on all data (no holdout — CV already measured it)
  5. save_and_register(...)      — write artifact + model_runs row

The model_runs table is the deployment mechanism: `load_model()` in predict.py
always picks the latest row by `trained_at`, so committing a new row is the
entire deploy step — no code change, no redeploy.

M5 gate: cv_mae must beat both baselines (roll10_pts and season_avg_pts).
A warning is logged but the pipeline does NOT abort when this fails, because
aborting would leave predictions stale. Diagnose the feature pipeline first.
"""

from __future__ import annotations

import logging
import sys
from datetime import date

logger = logging.getLogger(__name__)


_STATS = ("pts", "reb", "ast", "fg3m")


def run_retrain(as_of_date: date | None = None, *, n_trials: int = 50) -> dict[str, dict]:
    """Train one XGBoost model per stat and register all in model_runs.

    Parameters
    ----------
    as_of_date:
        Upper bound for training data. Defaults to today. Set explicitly
        in backtesting to prevent leakage from future data.
    n_trials:
        Optuna trial count. 50 is the production default; pass 5 for a
        fast smoke test (``--fast`` CLI flag).

    Returns
    -------
    Dict keyed by stat name, each value is the run_training summary dict.
    """
    from nba_predictor.model.train import run_training

    logger.info("=== run_retrain as_of=%s, n_trials=%d, stats=%s ===", as_of_date or "today", n_trials, _STATS)
    summaries: dict[str, dict] = {}
    for stat in _STATS:
        logger.info("--- Training stat=%s ---", stat)
        summaries[stat] = run_training(stat=stat, as_of_date=as_of_date, n_trials=n_trials)

    logger.info("run_retrain complete — cv_maes: %s", {s: f"{summaries[s]['cv_mae']:.4f}" for s in _STATS})
    return summaries


if __name__ == "__main__":
    # `uv run python -m nba_predictor.pipelines.run_retrain [--fast]`
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(name)s | %(message)s")

    fast = "--fast" in sys.argv
    results = run_retrain(n_trials=5 if fast else 50)

    print("\n=== Retrain complete ===")
    for stat, summary in results.items():
        print(f"\n  [{stat}]")
        for k, v in summary.items():
            print(f"    {k}: {v}")
