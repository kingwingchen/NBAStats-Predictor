# NBA Points Predictor

An automated, end-to-end ML pipeline that predicts each qualifying NBA player's points in their next game. Data flows from `nba_api` → Supabase Postgres → an XGBoost regressor (tuned with Optuna under walk-forward CV) → a public Streamlit dashboard. Daily ingest + inference and biweekly retrains run on GitHub Actions.

See [`PLAN.md`](./PLAN.md) for the full architecture, schema, feature set, and build-order milestones.

## Stack

- **Python 3.12** managed with [`uv`](https://docs.astral.sh/uv/)
- **Data:** `nba_api`, pandas
- **Storage:** Supabase (Postgres) via SQLAlchemy
- **Model:** XGBoost regression, Optuna tuning, walk-forward CV
- **Dashboard:** Streamlit (Community Cloud)
- **Automation:** GitHub Actions (daily + biweekly cron)

## Setup

Prerequisites: `uv`, a free Supabase project.

```bash
# 1. Install deps (creates .venv automatically)
uv sync --all-groups

# 2. Configure secrets
cp .env.example .env
# then paste your Supabase session-pooler URI into SUPABASE_DB_URL

# 3. Apply the database schema
#    Option A — Supabase SQL editor: paste src/nba_predictor/db/schema.sql
#    Option B — via any psql client pointed at SUPABASE_DB_URL

# 4. Verify the connection
uv run python -m nba_predictor.db.connection
# → should print the Postgres version string

# 5. (Optional) install pre-commit hooks
uv run pre-commit install
```

## Repo layout

```
src/nba_predictor/
  config.py            # env loading + project constants
  db/                  # SQLAlchemy engine + schema.sql (source of truth)
  ingest/              # nba_api wrapper + backfill/daily ingest
  features/            # leak-proof feature builder
  model/               # train / predict / evaluate
  pipelines/           # run_daily.py, run_retrain.py (orchestrators)
  app/                 # Streamlit dashboard
tests/                 # leak checks, idempotency, CV sanity
```

## Status

Phase 0 — scaffold + DB connection.
