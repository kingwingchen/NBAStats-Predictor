"""NBA Points Predictor — Streamlit dashboard.

Two-tab layout:
  Tab 1 — Tonight's Slate  : sortable prediction table for all qualifying players
                             on tonight's schedule, with model metadata footer.
  Tab 2 — Player Lookup    : searchable dropdown → prediction headline + 20-game
                             history chart + feature bar chart.

Run locally:
    uv run streamlit run src/nba_predictor/app/streamlit_app.py

Streamlit Community Cloud:
    Main file path : src/nba_predictor/app/streamlit_app.py
    Secret         : SUPABASE_DB_URL  (set in app settings → Secrets)
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nba_predictor.app.data import (
    compute_player_feature_snapshot,
    load_model_info,
    load_opp_season_stats,
    load_player_history,
    load_qualifying_players,
    load_residual_params_cached,
    load_tonights_slate,
)
from nba_predictor.model.calibration import compute_probability

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NBA Points Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Tab 1 — Tonight's Slate
# ---------------------------------------------------------------------------


def _render_slate_tab(slate_df: pd.DataFrame, model_info: dict | None) -> None:
    if slate_df.empty:
        st.info(
            "No predictions available for today. "
            "The daily ingest pipeline may not have run yet, or it's an off-season day."
        )
        return

    pred_date = slate_df["prediction_date"].iloc[0]
    n_games = slate_df["game_id"].nunique()
    st.subheader(f"Tonight's Slate — {pred_date}  ·  {n_games} game{'s' if n_games != 1 else ''}")

    df = slate_df.copy()
    # PRA is only meaningful when all three base models have been trained
    df["pra"] = df["predicted_pts"] + df["predicted_reb"] + df["predicted_ast"]

    display = df[
        ["full_name", "team", "opp",
         "predicted_pts", "predicted_reb", "predicted_ast", "predicted_fg3m", "pra",
         "roll10_pts", "rest_days", "home_away"]
    ].copy()
    display.columns = ["Player", "Team", "Opp", "Pts", "Reb", "Ast", "3PM", "PRA", "Roll-10", "Rest", "H/A"]
    display = display.reset_index(drop=True)
    display.index = range(1, len(display) + 1)

    st.dataframe(
        display,
        use_container_width=True,
        column_config={
            "Pts":     st.column_config.NumberColumn(format="%.1f"),
            "Reb":     st.column_config.NumberColumn(format="%.1f"),
            "Ast":     st.column_config.NumberColumn(format="%.1f"),
            "3PM":     st.column_config.NumberColumn(format="%.1f"),
            "PRA":     st.column_config.NumberColumn(format="%.1f"),
            "Roll-10": st.column_config.NumberColumn(format="%.1f"),
            "Rest":    st.column_config.NumberColumn(format="%d"),
        },
        height=min(800, 36 + 35 * len(display)),
    )

    if model_info:
        trained = str(model_info.get("trained_at", ""))[:10]
        cv_mae_val = model_info.get("cv_mae")
        cv_mae = f"{cv_mae_val:.2f}" if cv_mae_val else "—"
        st.caption(
            f"Model trained {trained} · Pts CV MAE {cv_mae} · "
            f"{len(display)} players on tonight's slate"
        )
    else:
        st.caption("No trained model found — run `uv run python -m nba_predictor.model.train` first.")


# ---------------------------------------------------------------------------
# Tab 2 — Player Lookup
# ---------------------------------------------------------------------------


def _render_player_tab(slate_df: pd.DataFrame) -> None:
    players_df = load_qualifying_players()

    if players_df.empty:
        st.info("No qualifying players found. Run the historical backfill first.")
        return

    player_options = {
        int(row.player_id): row.full_name
        for row in players_df.itertuples(index=False)
    }

    selected_id: int | None = st.selectbox(
        "Search for a player",
        options=list(player_options.keys()),
        format_func=lambda pid: player_options.get(pid, str(pid)),
    )

    if selected_id is None:
        return

    # ----- Stat selector + Vegas line input --------------------------------
    _STAT_LABELS = {
        "pts":  "Points",
        "reb":  "Rebounds",
        "ast":  "Assists",
        "fg3m": "3-Pointers Made",
        "pr":   "Pts + Reb",
        "pa":   "Pts + Ast",
        "ra":   "Reb + Ast",
        "pra":  "Pts + Reb + Ast",
    }
    _STAT_COMBOS: dict[str, tuple[str, ...]] = {
        "pr": ("pts", "reb"), "pa": ("pts", "ast"),
        "ra": ("reb", "ast"), "pra": ("pts", "reb", "ast"),
    }

    sel_col, line_col = st.columns([2, 1])
    with sel_col:
        selected_stat: str = st.selectbox(
            "Stat",
            options=list(_STAT_LABELS.keys()),
            format_func=lambda s: _STAT_LABELS[s],
        )
    with line_col:
        vegas_line_raw: float = st.number_input(
            "Vegas line (optional)",
            min_value=0.0, max_value=200.0,
            value=0.0, step=0.5,
            help="Enter a line to display Over/Under probability",
        )
    vegas_line: float | None = float(vegas_line_raw) if vegas_line_raw > 0 else None
    stat_label = _STAT_LABELS[selected_stat]

    # ----- Tonight's prediction for this player ----------------------------
    player_row: dict | None = None
    if not slate_df.empty:
        rows = slate_df[slate_df["player_id"] == selected_id]
        if not rows.empty:
            player_row = rows.iloc[0].to_dict()

    def _get_pred(row: dict | None) -> float | None:
        if row is None:
            return None
        if selected_stat in ("pts", "reb", "ast", "fg3m"):
            v = row.get(f"predicted_{selected_stat}")
            return float(v) if v is not None and not pd.isna(v) else None
        sub_vals = [row.get(f"predicted_{s}") for s in _STAT_COMBOS[selected_stat]]
        if all(v is not None and not pd.isna(v) for v in sub_vals):
            return float(sum(sub_vals))  # type: ignore[arg-type]
        return None

    pred_val = _get_pred(player_row)

    st.markdown(f"### {player_options.get(selected_id, selected_id)}")

    if player_row:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"Predicted {stat_label}", f"{pred_val:.1f}" if pred_val is not None else "—")
        col2.metric("Opponent", player_row["opp"])
        rest = player_row.get("rest_days")
        col3.metric("Rest Days", int(rest) if rest is not None else "—")
        col4.metric("Home/Away", player_row.get("home_away", "—"))
    else:
        st.info("Not on tonight's slate — showing recent history below.")

    # ----- Over/Under probability (only when Vegas line is entered) --------
    if vegas_line is not None and pred_val is not None:
        if selected_stat in _STAT_COMBOS:
            # Combine residual params assuming independence: σ_combo = √(Σσᵢ²)
            sub_params = [load_residual_params_cached(s) for s in _STAT_COMBOS[selected_stat]]
            if all(sd is not None for _, sd in sub_params):
                r_mean: float | None = sum((m or 0.0) for m, _ in sub_params)
                r_std: float | None  = float(sum(sd ** 2 for _, sd in sub_params) ** 0.5)
            else:
                r_mean, r_std = None, None
        else:
            r_mean, r_std = load_residual_params_cached(selected_stat)

        if r_std is not None:
            probs = compute_probability(pred_val, vegas_line, r_std, r_mean or 0.0)
            p_over, p_under = probs["p_over"], probs["p_under"]
            edge = probs["edge"]

            st.markdown("---")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.markdown(f"**Over {vegas_line}**")
                st.progress(float(p_over), text=f"{p_over * 100:.1f}%")
            with p_col2:
                st.markdown(f"**Under {vegas_line}**")
                st.progress(float(p_under), text=f"{p_under * 100:.1f}%")
            direction = "Over" if edge > 0 else "Under"
            st.caption(
                f"Predicted: {pred_val:.1f} {stat_label} · "
                f"Model edge: {abs(edge) * 100:.1f}% toward {direction}"
            )
            st.markdown("---")
        else:
            st.caption("Calibration unavailable — run retrain to enable probability estimates.")

    # ----- Pickfinder-style bar chart (replaces line chart) ----------------
    history = load_player_history(selected_id)
    if history.empty:
        st.warning("No game history found for this player.")
        return

    display_history = history.tail(20)

    # Compute bar heights for the selected stat (or combo sum)
    if selected_stat in ("pts", "reb", "ast", "fg3m"):
        bar_vals = display_history[selected_stat].fillna(0).to_numpy(dtype=float)
    else:
        bar_vals = sum(
            display_history[c].fillna(0) for c in _STAT_COMBOS[selected_stat]
        ).to_numpy(dtype=float)

    # Reference line: Vegas line > tonight's prediction > roll-10 average
    if vegas_line is not None:
        ref_line = vegas_line
        ref_label = f"Line: {ref_line:.1f}"
    elif pred_val is not None:
        ref_line = pred_val
        ref_label = f"Predicted: {ref_line:.1f}"
    else:
        ref_line = float(bar_vals[-10:].mean()) if len(bar_vals) >= 1 else 0.0
        ref_label = f"Avg: {ref_line:.1f}"

    # Green = went Over the reference line, red = stayed Under
    bar_colors = ["#2ecc71" if v > ref_line else "#e74c3c" for v in bar_vals]
    y_max = max(float(bar_vals.max()) * 1.3, ref_line * 1.3, 5.0) if len(bar_vals) else 10.0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=display_history["game_date"],
        y=bar_vals,
        marker_color=bar_colors,
        text=[f"{v:.1f}" for v in bar_vals],
        textposition="outside",
    ))
    fig.add_hline(
        y=ref_line,
        line_dash="dash",
        line_color="rgba(255,255,255,0.75)",
        annotation_text=ref_label,
        annotation_position="top right",
        annotation_font_color="rgba(255,255,255,0.75)",
    )
    fig.update_layout(
        title=f"Last 20 Games — {stat_label}  (green = over reference line)",
        xaxis_title="Date",
        yaxis_title=stat_label,
        yaxis={"range": [0, y_max]},
        height=380,
        margin={"t": 60, "b": 40, "l": 50, "r": 20},
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ----- Feature bar chart -----------------------------------------------
    opp_stats: dict = {}
    if player_row and player_row.get("opp_team_id"):
        opp_stats = load_opp_season_stats(int(player_row["opp_team_id"]))

    rest_days_val = int(player_row["rest_days"]) if player_row and player_row.get("rest_days") is not None else None
    is_home_val   = int(player_row["is_home"])   if player_row and player_row.get("is_home")   is not None else None

    snapshot = compute_player_feature_snapshot(
        history,
        stat=selected_stat,
        rest_days=rest_days_val,
        is_home=is_home_val,
        opp_stats=opp_stats,
    )

    if snapshot:
        feat_items = [(k, v) for k, v in snapshot.items() if not pd.isna(v)]
        feat_df = pd.DataFrame(feat_items, columns=["Feature", "Value"])

        def _category_color(name: str) -> str:
            if "opp" in name:
                return "#f0803c"   # orange — opponent context
            if name in {"rest_days", "is_home", "is_back_to_back", "is_cold_start", "games_played_season"}:
                return "#2ecc71"   # green — game context
            return "#3498db"       # blue — player form / usage

        feat_df["color"] = feat_df["Feature"].map(_category_color)

        bar_fig = go.Figure(go.Bar(
            x=feat_df["Value"],
            y=feat_df["Feature"],
            orientation="h",
            marker_color=feat_df["color"],
            text=feat_df["Value"].round(1).astype(str),
            textposition="outside",
        ))
        bar_fig.update_layout(
            title=f"Feature Snapshot — {stat_label}  (blue: form, green: context, orange: opponent)",
            height=max(320, 28 * len(feat_df) + 80),
            xaxis_title="Value",
            yaxis={"autorange": "reversed"},
            margin={"l": 180, "r": 80, "t": 60, "b": 30},
        )
        st.plotly_chart(bar_fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("🏀 NBA Player Prop Predictor")
    st.caption(
        "Predicts Pts, Reb, Ast, 3PM and combination props for tonight's qualifying players. "
        "Updated daily via GitHub Actions · powered by XGBoost + Supabase."
    )

    slate_df = load_tonights_slate()
    model_info = load_model_info()

    tab1, tab2 = st.tabs(["Tonight's Slate", "Player Lookup"])

    with tab1:
        _render_slate_tab(slate_df, model_info)

    with tab2:
        _render_player_tab(slate_df)


if __name__ == "__main__":
    main()
