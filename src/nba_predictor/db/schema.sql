-- Source of truth for the NBA predictor's Supabase schema.
-- Idempotent: safe to re-run. Apply via Supabase SQL editor or MCP.
--
-- Design notes (so a reader/interviewer can follow the "why"):
--   * Points-only v1, but column-additive: adding reb/ast/3pm targets later
--     is just ALTER TABLE, not a rebuild.
--   * Composite PKs on (player_id, game_id) and (team_id, game_id) make
--     idempotent upserts during daily ingest trivial (ON CONFLICT DO UPDATE).
--   * predictions stores `actual_pts NULL` so the daily job can backfill
--     yesterday's actuals in one UPDATE — powers the accuracy-over-time view.
--   * model_runs decouples model artifacts from code: Streamlit always loads
--     the latest row here, so retrains deploy by INSERT, not redeploy.

-- ───────────────────────────────────────────────────────────── raw tables

CREATE TABLE IF NOT EXISTS games (
    game_id       TEXT       PRIMARY KEY,
    game_date     DATE       NOT NULL,
    season        TEXT       NOT NULL,
    home_team_id  INTEGER    NOT NULL,
    away_team_id  INTEGER    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_games_date      ON games (game_date);
CREATE INDEX IF NOT EXISTS idx_games_season    ON games (season);

CREATE TABLE IF NOT EXISTS players (
    player_id  INTEGER  PRIMARY KEY,
    full_name  TEXT     NOT NULL,
    position   TEXT,
    is_active  BOOLEAN  NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS player_game_logs (
    player_id    INTEGER  NOT NULL,
    game_id      TEXT     NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    team_id      INTEGER  NOT NULL,
    min          NUMERIC,
    pts          INTEGER,
    reb          INTEGER,
    ast          INTEGER,
    stl          INTEGER,
    blk          INTEGER,
    tov          INTEGER,
    fgm          INTEGER,
    fga          INTEGER,
    fg3m         INTEGER,
    fg3a         INTEGER,
    ftm          INTEGER,
    fta          INTEGER,
    plus_minus   INTEGER,
    PRIMARY KEY (player_id, game_id)
);

CREATE INDEX IF NOT EXISTS idx_pgl_player  ON player_game_logs (player_id);
CREATE INDEX IF NOT EXISTS idx_pgl_game    ON player_game_logs (game_id);

CREATE TABLE IF NOT EXISTS team_game_logs (
    team_id      INTEGER  NOT NULL,
    game_id      TEXT     NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    def_rating   NUMERIC,
    off_rating   NUMERIC,
    pace         NUMERIC,
    PRIMARY KEY (team_id, game_id)
);

CREATE INDEX IF NOT EXISTS idx_tgl_team  ON team_game_logs (team_id);

-- ──────────────────────────────────────────────────── derived / operational

CREATE TABLE IF NOT EXISTS model_runs (
    run_id          SERIAL        PRIMARY KEY,
    trained_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    train_end_date  DATE          NOT NULL,
    cv_mae          NUMERIC       NOT NULL,
    baseline_mae    NUMERIC,
    params_json     JSONB,
    artifact_path   TEXT          NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_model_runs_trained_at  ON model_runs (trained_at DESC);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id    BIGSERIAL  PRIMARY KEY,
    model_run_id     INTEGER    NOT NULL REFERENCES model_runs(run_id),
    player_id        INTEGER    NOT NULL,
    game_id          TEXT       NOT NULL,
    prediction_date  DATE       NOT NULL,
    predicted_pts    NUMERIC    NOT NULL,
    actual_pts       INTEGER,   -- backfilled the morning after the game
    UNIQUE (model_run_id, player_id, game_id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_date    ON predictions (prediction_date DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_player  ON predictions (player_id);
