"""Central configuration: environment variables and project-wide constants.

All tunable knobs live here so pipelines and notebooks read from one source of
truth. `.env` is loaded at import time; missing required secrets raise early
rather than silently producing broken DB URLs downstream.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Repo root = parents[2] from src/nba_predictor/config.py
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Load .env from the repo root exactly once. override=False so real env vars
# (e.g. GitHub Actions secrets) take precedence over local .env.
load_dotenv(REPO_ROOT / ".env", override=False)


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required env var: {name}. Copy .env.example to .env and fill it in."
        )
    return value


# --- Secrets ---------------------------------------------------------------
SUPABASE_DB_URL: str = _require("SUPABASE_DB_URL")


# --- Project constants -----------------------------------------------------
# Five full NBA seasons of history. Labels match nba_api's expected format.
# Ordered oldest → newest for deterministic backfill.
SEASONS: tuple[str, ...] = (
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
)

# Player universe filter. A player qualifies if their MPG in the prior OR
# current season is >= this threshold. Chosen to capture rotation players
# (trades, role changes, sophomore breakouts) without flooding the slate
# with deep-bench names whose pts targets are near-zero and add noise.
MPG_THRESHOLD: float = 15.0

# Feature-engineering windows. Short window (5) picks up hot/cold streaks;
# long window (10) approximates stable form. Both are shifted by one game
# to prevent target leakage — see features/build.py.
ROLLING_WINDOWS: tuple[int, ...] = (5, 10)

# A player with fewer than this many season games has unstable rolling
# features. We keep these rows (XGBoost handles NaN) and flag them with
# is_cold_start so the model can learn to down-weight them.
COLD_START_THRESHOLD: int = 10

# Artifact locations for trained models. Metadata (CV MAE, params) lives
# in the model_runs table; only the binary itself lives on disk.
ARTIFACTS_DIR: Path = REPO_ROOT / "artifacts" / "models"
