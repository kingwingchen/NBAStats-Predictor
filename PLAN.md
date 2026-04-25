# NBA Predictor — Version 2 Roadmap

This document tracks planned work for v2. V1 (points-only, automated pipeline, Streamlit dashboard) is complete and live.

---

## V2 Goals

1. **Multi-stat prediction** — extend the pipeline to predict rebounds, assists, and all combination props
2. **Probability output** — for any predicted stat + Vegas line, display Over % / Under % instead of just a point estimate
3. **Vegas line integration** — manual user input in the Player Lookup tab so predictions can be benchmarked against the market
4. **Lower MAE** — continued model improvement through better features, expanded tuning, and per-stat model tuning

---

## Scope — V2 Features

### Stat targets

| Stat | Description |
|---|---|
| `pts` | Points (already live in v1) |
| `reb` | Total rebounds |
| `ast` | Assists |
| `fg3m` | 3-pointers made |
| `pr` | Points + Rebounds |
| `pa` | Points + Assists |
| `ra` | Rebounds + Assists |
| `pra` | Points + Rebounds + Assists |

Combination props (`pr`, `pa`, `ra`, `pra`) are derived by summing the individual model predictions — no separate model needed. The per-stat models must improve independently to improve the combo props.

### Probability output (Over / Under)

For each prediction, display:
```
Predicted: 22.4 pts
Vegas line: 21.5 (user input)
Over 21.5: 58%   Under 21.5: 42%
```

**Method:** fit a residual distribution (Gaussian) over the walk-forward CV validation errors for each stat. At inference time, shift the distribution to the predicted value and integrate above/below the Vegas line to get Over/Under probabilities.

This is statistically principled — the CV residuals tell us the model's actual error distribution, so the probability reflects real historical calibration rather than an arbitrary assumption.

### Vegas line input — Player Lookup tab

Add a number input widget to the Player Lookup tab in Streamlit:
- User selects a stat type (pts / reb / ast / fg3m / pr / pa / ra / pra)
- User inputs the Vegas line for that stat
- Dashboard displays: predicted value, Over %, Under %, and the implied edge vs. the line

If no line is entered, the probability display is hidden.

### UI display

```
[Stat selector: Points ▼]   [Vegas line: 21.5]

Predicted:  22.4 pts
─────────────────────────────
Over  21.5   58%  ████████████░░░░░░░░
Under 21.5   42%  ░░░░░░░░████████████
```

Clear color coding: green for the higher-probability side.

---

## Implementation plan

### Phase A — Multi-stat data + features
- Add `reb`, `ast`, `fg3m` to the `features/build.py` SQL query (columns already exist in `player_game_logs`)
- Add rolling windows and season averages for each new stat (mirror existing `pts` pattern)
- Add `season_avg_reb`, `season_avg_ast`, `roll5_reb`, `roll10_reb`, etc. to `X_COLS` or a per-stat feature set
- Update `db/schema.sql` with `ALTER TABLE predictions ADD COLUMN predicted_reb NUMERIC` etc.

### Phase B — Per-stat models
- Train one XGBoost model per stat target (`pts`, `reb`, `ast`, `fg3m`)
- Each model uses the same walk-forward CV + Optuna pipeline, tuned independently
- `model_runs` table gets a `stat` column to identify which target each row belongs to
- `load_model(stat="pts")` — inference always picks the latest registered model for that stat

### Phase C — Residual calibration + probability output
- After walk-forward CV, collect all validation residuals per stat
- Fit `scipy.stats.norm` to the residual distribution per stat
- Store `(residual_mean, residual_std)` alongside the model artifact in `model_runs`
- At inference: `P(Over line) = 1 - norm.cdf(line, loc=predicted, scale=residual_std)`

### Phase D — Dashboard updates
- Tonight's Slate: add Reb, Ast, 3PM columns; PRA as a derived column
- Player Lookup: stat selector dropdown + Vegas line number input
- Over/Under probability display with progress-bar style visualization
- Feature bar chart adapts to the selected stat (shows reb-specific features when viewing rebounds)

---

## Deferred beyond v2

- Injury data / questionable tags
- Vegas line auto-ingestion (no free real-time API)
- Non-XGBoost models (LightGBM, neural nets)
- MLflow experiment tracking
- Player-vs-player matchup features
