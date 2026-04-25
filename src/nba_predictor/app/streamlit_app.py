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
    load_tonights_slate,
)

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

    display = slate_df[
        ["full_name", "team", "opp", "predicted_pts", "roll10_pts", "rest_days", "home_away"]
    ].copy()
    display.columns = ["Player", "Team", "Opponent", "Predicted Pts", "Roll-10 Avg", "Rest Days", "H/A"]
    display = display.reset_index(drop=True)
    display.index = range(1, len(display) + 1)

    st.dataframe(
        display,
        use_container_width=True,
        column_config={
            "Predicted Pts": st.column_config.NumberColumn(format="%.1f"),
            "Roll-10 Avg":   st.column_config.NumberColumn(format="%.1f"),
            "Rest Days":     st.column_config.NumberColumn(format="%d"),
        },
        # Cap height at 800px; 36px header + 35px per row is a reasonable estimate
        height=min(800, 36 + 35 * len(display)),
    )

    # Model metadata footer
    if model_info:
        trained = str(model_info.get("trained_at", ""))[:10]
        cv_mae_val = model_info.get("cv_mae")
        cv_mae = f"{cv_mae_val:.2f}" if cv_mae_val else "—"
        st.caption(
            f"Model trained {trained} · CV MAE {cv_mae} pts · "
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

    # ----- Tonight's prediction for this player ----------------------------
    player_row: dict | None = None
    if not slate_df.empty:
        rows = slate_df[slate_df["player_id"] == selected_id]
        if not rows.empty:
            player_row = rows.iloc[0].to_dict()

    if player_row:
        st.markdown(f"### {player_options.get(selected_id, selected_id)}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Predicted Pts", f"{player_row['predicted_pts']:.1f}")
        col2.metric("Opponent", player_row["opp"])
        rest = player_row.get("rest_days")
        col3.metric("Rest Days", int(rest) if rest is not None else "—")
        col4.metric("Home/Away", player_row.get("home_away", "—"))
    else:
        st.markdown(f"### {player_options.get(selected_id, selected_id)}")
        st.info("Not on tonight's slate — showing recent history below.")

    # ----- History line chart ----------------------------------------------
    history = load_player_history(selected_id)
    if history.empty:
        st.warning("No game history found for this player.")
        return

    history = history.copy()
    # Trailing 10-game rolling average for the chart — includes current game
    # (intuitive for visualization, distinct from the model's shift(1) feature)
    history["roll10_avg"] = history["pts"].rolling(10, min_periods=1).mean().round(1)
    display_history = history.tail(20)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=display_history["game_date"],
        y=display_history["pts"],
        mode="markers+lines",
        name="Actual Pts",
        marker={"size": 7},
        line={"width": 1.5},
    ))
    fig.add_trace(go.Scatter(
        x=display_history["game_date"],
        y=display_history["roll10_avg"],
        mode="lines",
        name="10-Game Avg",
        line={"dash": "dash", "width": 2},
    ))
    fig.update_layout(
        title="Last 20 Games",
        xaxis_title="Date",
        yaxis_title="Points",
        legend={"orientation": "h", "y": 1.15},
        height=350,
        margin={"t": 60, "b": 40, "l": 50, "r": 20},
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
        rest_days=rest_days_val,
        is_home=is_home_val,
        opp_stats=opp_stats,
    )

    if snapshot:
        # Drop NaN entries — features with no data (e.g. opp stats if not playing)
        feat_items = [(k, v) for k, v in snapshot.items() if not pd.isna(v)]
        feat_df = pd.DataFrame(feat_items, columns=["Feature", "Value"])

        # Colour-code by feature category so the chart is easier to scan
        def _category_color(name: str) -> str:
            if "opp" in name:
                return "#f0803c"   # orange — opponent context
            if name in {"rest_days", "is_home", "is_back_to_back", "is_cold_start", "games_played_season"}:
                return "#2ecc71"   # green — game context
            return "#3498db"       # blue — player scoring/usage

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
            title="Feature Snapshot  (model input values — blue: scoring, green: context, orange: opponent)",
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
    st.title("🏀 NBA Points Predictor")
    st.caption(
        "Predicts each qualifying player's points for tonight's games. "
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
