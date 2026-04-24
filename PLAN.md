# NBA Points Predictor — Master Blueprint

## One-liner
An automated, end-to-end ML pipeline that predicts each qualifying NBA player's points in their next game, refreshed daily via GitHub Actions, served through a public Streamlit dashboard, and built on a leak-proof walk-forward backtest.

---

## Architecture (data flow)

```
            ┌─────────────────┐
            │  nba_api        │  (PlayerGameLogs, TeamGameLogs, scoreboard)
            └────────┬────────┘
                     │  daily pull (GH Actions, ~9am ET)
                     ▼
            ┌─────────────────┐
            │  Supabase       │  raw tables: games, player_game_logs, team_game_logs
            │  (Postgres)     │  derived: player_features, predictions, model_runs
            └────────┬────────┘
                     │  SQL + pandas
                     ▼
            ┌─────────────────┐
            │  Feature builder │  rolling 5/10, season avg, rest, opp DRtg, is_cold_start
            └────────┬────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
 ┌──────────────┐          ┌──────────────┐
 │  Trainer     │          │  Inference    │
 │  (biweekly)  │          │  (daily)      │
 │  XGBoost +   │          │  load model,  │
 │  Optuna +    │──model──▶│  score tonight│
 │  walk-fwd CV │  artifact│  slate        │
 └──────────────┘          └──────┬───────┘
                                  │
                                  ▼
                          ┌──────────────┐
                          │  Streamlit    │  (Community Cloud, public)
                          │  - Tonight's  │
                          │    slate      │
                          │  - Player     │
                          │    lookup     │
                          └──────────────┘
```

---

## Repo structure

```
NBAStats-Predictor/
├── pyproject.toml              # uv-managed
├── uv.lock
├── .env.example                # SUPABASE_DB_URL, etc.
├── .github/workflows/
│   ├── daily_ingest.yml        # cron: ingest + inference
│   └── biweekly_retrain.yml    # cron: retrain + deploy artifact
├── src/nba_predictor/
│   ├── __init__.py
│   ├── config.py               # env loading, constants (SEASONS, MPG_THRESHOLD)
│   ├── db/
│   │   ├── connection.py       # SQLAlchemy engine factory
│   │   ├── schema.sql          # DDL — source of truth
│   │   └── models.py           # SQLAlchemy models (optional, for typed queries)
│   ├── ingest/
│   │   ├── backfill.py         # one-time 5-season historical pull
│   │   ├── daily.py            # yesterday's games → DB
│   │   └── nba_client.py       # nba_api wrapper with retry/rate-limit
│   ├── features/
│   │   ├── build.py            # the one function: (as_of_date) → feature DataFrame
│   │   └── player_universe.py  # resolve ≥15 MPG filter
│   ├── model/
│   │   ├── train.py            # walk-forward CV + Optuna + persist artifact
│   │   ├── predict.py          # load artifact + predict on feature DF
│   │   └── evaluate.py         # MAE, baselines, plots
│   ├── pipelines/
│   │   ├── run_daily.py        # orchestrates: ingest → features → predict → write
│   │   └── run_retrain.py      # orchestrates: features(hist) → train → persist
│   └── app/
│       └── streamlit_app.py    # dashboard entry point
├── artifacts/
│   └── models/                 # xgb_<timestamp>.json + metadata.json
├── notebooks/
│   └── 01_eda.ipynb            # kept minimal, for README screenshots
├── tests/
│   ├── test_features.py        # leak check: no future info in any row
│   ├── test_ingest.py          # idempotency check
│   └── test_model.py           # walk-forward CV produces non-trivial MAE
└── README.md                   # kept from Part 1, fleshed out progressively (interview centerpiece)
```

---

## Database schema (Supabase)

Six tables. Designed so adding rebounds/assists later is a column add, not a rebuild.

```sql
-- Raw from nba_api
games (
  game_id TEXT PRIMARY KEY,
  game_date DATE NOT NULL,
  season TEXT NOT NULL,
  home_team_id INT NOT NULL,
  away_team_id INT NOT NULL
);

player_game_logs (
  player_id INT NOT NULL,
  game_id TEXT NOT NULL REFERENCES games(game_id),
  team_id INT NOT NULL,
  min NUMERIC, pts INT, reb INT, ast INT, fg3m INT,
  fga INT, fta INT, tov INT, plus_minus INT,
  -- full playergamelogs column set
  PRIMARY KEY (player_id, game_id)
);

team_game_logs (
  team_id INT, game_id TEXT REFERENCES games(game_id),
  def_rating NUMERIC, off_rating NUMERIC, pace NUMERIC,
  PRIMARY KEY (team_id, game_id)
);

players (
  player_id INT PRIMARY KEY,
  full_name TEXT, position TEXT, is_active BOOLEAN
);

-- Derived / operational
predictions (
  prediction_id BIGSERIAL PRIMARY KEY,
  model_run_id INT REFERENCES model_runs(run_id),
  player_id INT, game_id TEXT,
  prediction_date DATE,         -- "as of" date
  predicted_pts NUMERIC,
  actual_pts INT NULL           -- backfilled next day
);

model_runs (
  run_id SERIAL PRIMARY KEY,
  trained_at TIMESTAMPTZ,
  train_end_date DATE,          -- last date in training set
  cv_mae NUMERIC, baseline_mae NUMERIC,
  params_json JSONB,
  artifact_path TEXT
);
```

Indices on `player_game_logs(player_id, game_id)`, `games(game_date)`, `predictions(prediction_date)`.

---

## Feature set (v1)

From `playergamelogs` + opponent joins, all computed with a `shift(1)` to prevent leakage:

| Feature | Definition |
|---|---|
| `roll5_pts`, `roll10_pts` | Mean points over last 5 / 10 games played (before current game) |
| `roll5_min`, `roll10_min` | Same for minutes |
| `roll5_fga`, `roll10_fga` | Same for field goal attempts (usage proxy) |
| `season_avg_pts` | Cumulative mean points this season, excluding current game |
| `season_avg_min` | Same for minutes |
| `games_played_season` | Count before current game |
| `rest_days` | Days since player's previous game (capped at 7) |
| `is_back_to_back` | 1 if rest_days == 1 |
| `is_home` | 1 if player's team == home team for this game |
| `opp_def_rating_roll10` | Opponent's rolling 10-game defensive rating |
| `opp_pace_roll10` | Opponent's rolling 10-game pace |
| `is_cold_start` | 1 if `games_played_season` < 10 |
| **Target:** `pts` | Points scored in current game |

XGBoost handles NaNs natively, so the first ~10 games of each player's season are kept, not dropped.

---

## Modeling protocol

**Split:** Walk-forward, expanding window. For each fold:
- Train on all games from 2020-21 season start through date `T`
- Validate on all games from `T+1` to `T+30 days`
- Slide `T` forward 30 days, refit, repeat
- Report **mean CV MAE across folds** as the headline metric

**Baselines (must beat both):**
1. `season_avg_pts` (naive persistence)
2. `roll10_pts` (short memory)

If XGBoost doesn't beat both, the features are broken — diagnose before tuning.

**Hyperparameter tuning:** Optuna, 50 trials, pruning enabled, search space:
```python
{
  "max_depth": (3, 10),
  "learning_rate": log-uniform(0.01, 0.3),
  "n_estimators": (100, 2000) with early_stopping,
  "min_child_weight": (1, 10),
  "subsample": (0.6, 1.0),
  "colsample_bytree": (0.6, 1.0),
  "reg_alpha": log-uniform(1e-3, 10),
  "reg_lambda": log-uniform(1e-3, 10),
}
```
Objective: mean walk-forward CV MAE (not a single holdout).

---

## Automation cadence

| Workflow | Schedule | Action |
|---|---|---|
| `daily_ingest.yml` | `0 13 * * *` (9am ET) | Pull yesterday's games → backfill `actual_pts` for last prediction → build features for today's slate → run inference → write to `predictions` |
| `biweekly_retrain.yml` | `0 14 * * MON/14` (every other Mon) | Rebuild full feature set → run Optuna + walk-forward CV → persist new artifact → write row to `model_runs` → Streamlit auto-picks latest |

Secrets: `SUPABASE_DB_URL` in GitHub repo secrets.

---

## Streamlit dashboard (two tabs)

**Tab 1 — Tonight's slate**
- Table: Player | Team | Opponent | Predicted Pts | Rolling-10 | Rest | Home/Away
- Sortable; default sort by predicted_pts desc
- Small "model info" footer: train date, CV MAE, # features

**Tab 2 — Player lookup**
- Searchable dropdown of qualifying players
- Headline: next-game prediction + date + opponent
- Line chart: last 20 games actual vs. rolling-10 avg
- Bar chart: top-10 feature values for tonight's prediction (not SHAP in v1 — just raw values; SHAP is a future-work upgrade)

---

## Build order & milestones

Tasks are tracked in the session task list. Milestone gates:

- **M1 (end of Phase 1):** `SELECT COUNT(*) FROM player_game_logs` returns ~100k rows across 5 seasons. Universe query returns ~150–250 players.
- **M2 (end of Phase 2):** `test_features.py` passes — no row's feature references a game on or after its own date.
- **M3 (end of Phase 3):** Walk-forward CV MAE beats both baselines by a clear margin (target: ≤5.5 pts MAE vs. ~6.5 for `roll10`).
- **M4 (end of Phase 4):** Predictions table has a row for every qualifying player on today's slate.
- **M5 (end of Phase 5):** Streamlit app is public-accessible, loads in under 3s.
- **M6 (end of Phase 6):** Two consecutive days of green workflow runs without manual intervention.

### Phases

- **Phase 0 — Project scaffold:** `uv init`, repo structure, Supabase project + connection, `.env` template, secrets management, pre-commit + ruff/black. No data yet.
- **Phase 1 — Historical backfill:** Ingest 5 seasons of PlayerGameLogs into Supabase. Idempotent upserts keyed on (player_id, game_id). Resolve ≥15 MPG player universe.
- **Phase 2 — Feature engineering layer:** Build SQL/pandas features per the table above. Leak-proof (shift-before-rolling).
- **Phase 3 — Model training + walk-forward CV + Optuna:** XGBoost regressor, expanding-window walk-forward CV, Optuna hyperparam search, MAE + baseline comparison, save artifact + metrics to disk.
- **Phase 4 — Daily inference pipeline:** Script that (1) ingests yesterday's games, (2) finds tonight's qualifying players, (3) builds features as-of today, (4) runs model, (5) writes predictions table.
- **Phase 5 — Streamlit dashboard:** Two-tab app per spec above. Deploy to Streamlit Community Cloud.
- **Phase 6 — GitHub Actions automation:** Daily ingest+inference workflow, biweekly retrain workflow, both with secrets-based DB auth. Badges + README.

---

## Scope — explicitly deferred (v2 roadmap)

Documented so scope creep is visible: **injury data ingestion, Vegas lines, multi-stat targets (reb/ast/3PM), SHAP explanations, prediction intervals, MLflow, player news sentiment.**

Reasoning for each is in conversation history; don't silently re-add without discussing.
