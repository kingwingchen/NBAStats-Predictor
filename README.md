# NBA Points Predictor

[![Daily Ingest](https://github.com/kingwingchen/NBAStats-Predictor/actions/workflows/daily_ingest.yml/badge.svg)](https://github.com/kingwingchen/NBAStats-Predictor/actions/workflows/daily_ingest.yml)
[![Biweekly Retrain](https://github.com/kingwingchen/NBAStats-Predictor/actions/workflows/biweekly_retrain.yml/badge.svg)](https://github.com/kingwingchen/NBAStats-Predictor/actions/workflows/biweekly_retrain.yml)

An automated, end-to-end ML pipeline that predicts each qualifying NBA player's points for their next game. Raw game logs flow from `nba_api` → Supabase Postgres → feature engineering → XGBoost (tuned with Optuna under walk-forward CV) → a public Streamlit dashboard. Daily ingest + inference and biweekly retrains run fully unattended on GitHub Actions.

**Live dashboard:** *(deploy to Streamlit Community Cloud — link here once deployed)*

---

## Architecture

```
nba_api  ──daily pull──▶  Supabase Postgres
                          (games, player_game_logs, team_game_logs)
                                    │
                              SQL + pandas
                                    │
                          Feature builder
                          (13 rolling/context features,
                           shift(1) leak-proof)
                                    │
               ┌────────────────────┴────────────────────┐
               ▼                                         ▼
        Trainer (biweekly)                    Inference (daily)
        XGBoost + Optuna +                    load latest model,
        walk-forward CV                       score tonight's slate
               │ artifact                             │
               └──────── model_runs (DB) ─────────────┘
                                    │
                          Streamlit dashboard
                          (Tonight's Slate + Player Lookup)
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.12, managed with [uv](https://docs.astral.sh/uv/) |
| Data ingestion | `nba_api` — player & team game logs, scoreboard |
| Storage | Supabase (Postgres) via SQLAlchemy |
| Feature engineering | pandas — rolling windows, expanding season averages |
| Model | XGBoost regression · Optuna hyperparameter search |
| Validation | Walk-forward expanding CV, MAE metric |
| Dashboard | Streamlit (Community Cloud) + Plotly |
| Automation | GitHub Actions — daily cron + biweekly retrain |

---

## Feature set

All features are computed with `shift(1)` inside player groups — no feature at row N contains information from game N, preventing data leakage.

| Feature | Description |
|---|---|
| `roll5_pts`, `roll10_pts` | Rolling mean points over last 5 / 10 games |
| `roll5_min`, `roll10_min` | Rolling mean minutes (usage proxy) |
| `roll5_fga`, `roll10_fga` | Rolling mean field-goal attempts |
| `season_avg_pts`, `season_avg_min` | Cumulative season average (resets each season) |
| `games_played_season` | Games played so far this season |
| `rest_days` | Days since previous game (capped at 7) |
| `is_back_to_back` | 1 if rest_days == 1 |
| `is_home` | 1 if player's team is the home team |
| `opp_def_rating_roll10` | Opponent's season defensive rating |
| `opp_pace_roll10` | Opponent's season pace |
| `is_cold_start` | 1 if fewer than 10 season games played |

**Baselines the model must beat:** `roll10_pts` MAE and `season_avg_pts` MAE. If XGBoost doesn't outperform both, the feature pipeline is diagnosed before any hyperparameter tuning.

---

## Project layout

```
src/nba_predictor/
  config.py                 # env vars + project constants
  db/
    connection.py           # SQLAlchemy engine factory (singleton, pool_pre_ping)
    schema.sql              # DDL source of truth — idempotent CREATE IF NOT EXISTS
  ingest/
    nba_client.py           # Retrying, rate-limited nba_api wrapper
    backfill.py             # One-time 5-season historical pull
    daily.py                # Yesterday's games → DB + backfill actuals
    players.py              # players table from CommonAllPlayers
    team_stats.py           # team_game_logs from LeagueDashTeamStats
  features/
    build.py                # build_features(as_of_date) → leak-proof DataFrame
    player_universe.py      # ≥15 MPG filter (current + prior season)
  model/
    train.py                # walk_forward_cv + Optuna + save_and_register
    predict.py              # load_model (from model_runs) + predict
    evaluate.py             # MAE + baseline comparison
  pipelines/
    run_daily.py            # Orchestrates daily ingest → inference → predictions
    run_retrain.py          # Orchestrates biweekly retrain → artifact
  app/
    data.py                 # Cached SQL data layer for Streamlit
    streamlit_app.py        # Two-tab dashboard: Tonight's Slate + Player Lookup
.github/workflows/
  daily_ingest.yml          # Cron 9am ET: run_daily
  biweekly_retrain.yml      # Cron even Mondays 10am ET: run_retrain
tests/
  conftest.py               # Load .env before test collection
  test_features.py          # Leak-proof invariants (requires DB)
  test_ingest.py            # Upsert idempotency (requires DB)
  test_model.py             # Walk-forward CV sanity on synthetic data (no DB)
```

---

## Setup

**Prerequisites:** [uv](https://docs.astral.sh/uv/), a free [Supabase](https://supabase.com/) project.

```bash
# 1. Clone and install
git clone https://github.com/kingwingchen/NBAStats-Predictor.git
cd NBAStats-Predictor
uv sync --all-groups

# 2. Configure secrets
cp .env.example .env
# Edit .env: paste your Supabase session-pooler URI into SUPABASE_DB_URL

# 3. Apply the database schema (idempotent — safe to re-run)
uv run python scripts/apply_schema.py

# 4. Verify the connection
uv run python -m nba_predictor.db.connection
# → prints the Postgres version string

# 5. (Optional) install pre-commit hooks
uv run pre-commit install
```

---

## Running the pipeline manually

```bash
# --- Historical backfill (one-time setup) ---
# Ingest 5 seasons of player + team game logs (~100k rows, takes ~10 min)
uv run python -m nba_predictor.ingest.players
uv run python -m nba_predictor.ingest.backfill
uv run python -m nba_predictor.ingest.team_stats

# --- Train the model ---
uv run python -m nba_predictor.model.train
# (add --fast for a 5-trial smoke test)

# --- Run today's inference ---
uv run python -m nba_predictor.pipelines.run_daily

# --- Launch the dashboard ---
uv run streamlit run src/nba_predictor/app/streamlit_app.py
```

---

## Automation (GitHub Actions)

| Workflow | Schedule | What it does |
|---|---|---|
| `daily_ingest.yml` | 9 AM ET every day | Ingest yesterday's results → backfill `actual_pts` → build inference features → score tonight's slate → write predictions |
| `biweekly_retrain.yml` | 10 AM ET even Mondays | Full feature rebuild → Optuna (50 trials) + walk-forward CV → final model on all data → register in `model_runs` |

**GitHub secret required:** `SUPABASE_DB_URL`

Both workflows support `workflow_dispatch` for manual runs. The retrain workflow exits with code 1 if the new model doesn't beat the `roll10_pts` baseline, surfacing regressions before they reach production.

---

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) → "New app".
3. Set **Main file path** to `src/nba_predictor/app/streamlit_app.py`.
4. Under **Secrets**, add `SUPABASE_DB_URL = "your-connection-string"`.
5. Deploy — Streamlit auto-picks the latest model from the `model_runs` table on each page load.

---

## ML methodology notes

**Why walk-forward CV instead of random split?**
NBA game logs are time-series data. A random split lets rows from January train on rows from February (future leakage). Walk-forward expanding CV strictly trains on all games before date T and validates on T+1..T+30, matching the real-world prediction scenario.

**Why MAE over RMSE?**
NBA points distributions have occasional outliers (40+ point games). RMSE squares those, causing the model to over-fit to rare blow-up performances. MAE treats every point equally and is more interpretable ("on average, we're off by X points").

**Why XGBoost over a neural network?**
With ~100k training rows and 13 features, gradient-boosted trees consistently match or outperform neural networks on tabular data at this scale, train in minutes rather than hours, and produce interpretable feature importances — all critical for a portfolio project that needs to be explained in interviews.

**Cold-start handling:**
Rather than dropping a player's first 10 games (where rolling averages are unstable), we flag them with `is_cold_start = 1` and let XGBoost learn a separate leaf partition. This retains all data while letting the model down-weight early-season predictions automatically.
