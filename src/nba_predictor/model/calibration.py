"""Probability calibration for Over/Under predictions.

Given a model prediction and a Vegas prop line, returns the probability the
player exceeds (Over) or misses (Under) the line, using the residual
distribution fit during walk-forward CV.

Theory
------
residual = actual − predicted  →  actual ~ N(predicted + residual_mean, residual_std²)

P(actual > line) = 1 − norm.cdf(line, loc = predicted + residual_mean, scale = residual_std)

If the model is unbiased (residual_mean ≈ 0), this simplifies to centering
the Gaussian directly on the prediction.  residual_mean is stored in model_runs
for diagnostics and applied here for proper bias correction.
"""

from __future__ import annotations

import logging

from scipy.stats import norm
from sqlalchemy import text

from nba_predictor.db.connection import get_engine

logger = logging.getLogger(__name__)


def compute_probability(
    predicted: float,
    line: float,
    residual_std: float,
    residual_mean: float = 0.0,
) -> dict[str, float]:
    """Return Over/Under probabilities for a predicted stat vs. a Vegas prop line.

    Parameters
    ----------
    predicted:
        Model's point estimate for the stat.
    line:
        The Vegas prop line to compare against.
    residual_std:
        Standard deviation of CV residuals stored in model_runs. Controls the
        width of the probability distribution — larger std = less certainty.
    residual_mean:
        Mean of CV residuals (model bias). Defaults to 0.0. Non-zero values
        shift the Gaussian center to correct for systematic over/under-prediction.

    Returns
    -------
    {"p_over": float, "p_under": float, "edge": float}
        p_over + p_under == 1.0 by construction.
        edge = p_over − 0.5: positive means lean Over, negative means lean Under.
    """
    center = predicted + residual_mean
    p_over = float(1.0 - norm.cdf(line, loc=center, scale=residual_std))
    p_under = 1.0 - p_over
    return {
        "p_over": round(p_over, 4),
        "p_under": round(p_under, 4),
        "edge": round(p_over - 0.5, 4),
    }


def load_residual_params(stat: str = "pts") -> tuple[float, float] | tuple[None, None]:
    """Fetch (residual_mean, residual_std) for the latest trained model for `stat`.

    Returns (None, None) if no model has been trained for this stat yet, or
    if the model_runs row pre-dates Phase C (no residual columns recorded).
    """
    with get_engine().connect() as conn:
        row = conn.execute(
            text(
                "SELECT residual_mean, residual_std FROM model_runs "
                "WHERE stat = :stat AND residual_std IS NOT NULL "
                "ORDER BY trained_at DESC LIMIT 1"
            ),
            {"stat": stat},
        ).fetchone()

    if row is None:
        logger.debug("No residual params found for stat=%s — P(Over/Under) unavailable", stat)
        return None, None

    return float(row.residual_mean), float(row.residual_std)
