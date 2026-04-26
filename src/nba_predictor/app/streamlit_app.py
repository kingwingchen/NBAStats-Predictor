"""NBA Player Prop Predictor — Streamlit dashboard.

Two-tab layout:
  Tab 1 — Tonight's Slate  : Pickfinder-style hit-rate table for all qualifying
                             players on tonight's schedule. Enter a Vegas line to
                             see color-coded L5/L10/L15/L30/H2H/SZN columns.
  Tab 2 — Player Lookup    : searchable dropdown → prediction headline +
                             Pickfinder hit-rate tiles + window-selectable bar
                             chart + feature snapshot chart.

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
    NBA_TEAMS,
    compute_player_feature_snapshot,
    load_bulk_player_histories,
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
# Shared constants
# ---------------------------------------------------------------------------

_STAT_LABELS: dict[str, str] = {
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
    "pr":  ("pts", "reb"),
    "pa":  ("pts", "ast"),
    "ra":  ("reb", "ast"),
    "pra": ("pts", "reb", "ast"),
}

_HIT_RATE_COLS = ["L5", "L10", "L15", "L30", "H2H", "SZN"]

_WINDOW_SIZES: dict[str, int] = {
    "L5": 5, "L10": 10, "L15": 15, "L20": 20, "L30": 30,
}

# Sorted alphabetically by abbreviation; "— Any —" disables H2H filtering.
_TEAM_OPTIONS: dict[str, int | None] = {
    "— Any —": None,
    **{abbr: tid for tid, abbr in sorted(NBA_TEAMS.items(), key=lambda x: x[1])},
}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_pred_val(row: dict, stat: str) -> float | None:
    """Extract the predicted value for `stat` from a slate row dict."""
    if stat in _STAT_COMBOS:
        parts = [row.get(f"predicted_{s}") for s in _STAT_COMBOS[stat]]
        if all(v is not None and not pd.isna(v) for v in parts):
            return round(float(sum(parts)), 1)  # type: ignore[arg-type]
        return None
    v = row.get(f"predicted_{stat}")
    return round(float(v), 1) if v is not None and not pd.isna(v) else None


def compute_hit_rates(
    history_df: pd.DataFrame,
    stat: str,
    line: float,
    opp_team_id: int | None = None,
) -> dict[str, float | None]:
    """Return % of recent games the player went Over `line` for `stat`.

    Windows L5/L10/L15/L30 use the most-recent N games; SZN uses all
    available history (up to 35 rows); H2H filters to games against tonight's
    specific opponent. Returns None for any window with zero qualifying games.
    """
    empty: dict[str, float | None] = {k: None for k in _HIT_RATE_COLS}
    if history_df.empty:
        return empty

    if stat in _STAT_COMBOS:
        cols = _STAT_COMBOS[stat]
        vals: pd.Series = sum(history_df[c].fillna(0) for c in cols)  # type: ignore[assignment]
    else:
        if stat not in history_df.columns:
            return empty
        vals = history_df[stat].fillna(0)

    def _pct(s: pd.Series) -> float | None:
        if len(s) == 0:
            return None
        return round(float((s > line).sum()) / len(s) * 100, 1)

    result: dict[str, float | None] = {
        "L5":  _pct(vals.tail(5)),
        "L10": _pct(vals.tail(10)),
        "L15": _pct(vals.tail(15)),
        "L30": _pct(vals.tail(30)),
        "SZN": _pct(vals),
    }
    if opp_team_id is not None and "opp_team_id" in history_df.columns:
        mask = history_df["opp_team_id"] == opp_team_id
        result["H2H"] = _pct(vals[mask])
    else:
        result["H2H"] = None
    return result


def _pct_bg(val: object) -> str:
    """Map a 0–100 hit-rate to a CSS background-color hex string."""
    try:
        v = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "#2a2a2a"
    if pd.isna(v):
        return "#2a2a2a"
    if v >= 75:
        return "#1a5e35"
    if v >= 60:
        return "#27ae60"
    if v >= 50:
        return "#2874a6"
    if v >= 40:
        return "#c0392b"
    return "#7b241c"


def _pct_css(val: object) -> str:
    """Full CSS string (background + foreground) for a Styler cell callback."""
    try:
        v = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    if pd.isna(v):
        return ""
    return f"background-color: {_pct_bg(val)}; color: #ffffff"


def _hit_rate_html(rates: dict[str, float | None], line: float, label: str) -> str:
    """Build a Pickfinder-style colored HTML tile row for one player's hit rates."""
    header_cells = "".join(
        f'<th style="text-align:center; color:#aaaaaa; font-size:0.78em; '
        f'padding:4px 12px; font-weight:600; letter-spacing:0.05em;">'
        f'{k}</th>'
        for k in _HIT_RATE_COLS
    )
    data_cells = ""
    for k in _HIT_RATE_COLS:
        val = rates.get(k)
        text = f"{val:.0f}%" if val is not None else "—"
        bg = _pct_bg(val)
        data_cells += (
            f'<td style="text-align:center; font-size:1.25em; font-weight:700; '
            f'color:#ffffff; background:{bg}; padding:14px 12px; '
            f'border-radius:6px; min-width:60px;">{text}</td>'
        )
    return f"""
<p style="color:#aaaaaa; font-size:0.85em; margin:12px 0 6px 0;">
  Hit Rate vs <strong style="color:#ffffff;">{line}</strong> line — {label}
</p>
<table style="width:100%; border-collapse:separate; border-spacing:5px 0; margin-bottom:16px;">
  <tr>{header_cells}</tr>
  <tr>{data_cells}</tr>
</table>
"""


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
    st.subheader(
        f"Tonight's Slate — {pred_date}  ·  {n_games} game{'s' if n_games != 1 else ''}"
    )

    sel_col, line_col = st.columns([2, 1])
    with sel_col:
        selected_stat: str = st.selectbox(
            "Stat",
            options=list(_STAT_LABELS.keys()),
            format_func=lambda s: _STAT_LABELS[s],
            key="slate_stat",
        )
    with line_col:
        vegas_line_raw: float = st.number_input(
            "Vegas line (optional)",
            min_value=0.0, max_value=200.0,
            value=0.0, step=0.5,
            key="slate_line",
            help="Enter a prop line to see Pickfinder-style hit-rate columns",
        )
    vegas_line: float | None = float(vegas_line_raw) if vegas_line_raw > 0 else None

    df = slate_df.copy()
    df["Predicted"] = df.apply(
        lambda r: _get_pred_val(r.to_dict(), selected_stat), axis=1
    )

    display = df[["full_name", "team", "opp", "home_away", "rest_days", "Predicted"]].copy()
    display.columns = ["Player", "Team", "Opp", "H/A", "Rest", "Predicted"]
    display = display.reset_index(drop=True)
    display.index = range(1, len(display) + 1)

    if vegas_line is not None:
        # One SQL round-trip for all players — avoids N+1 query problem
        player_ids: tuple[int, ...] = tuple(int(pid) for pid in df["player_id"])
        all_histories = load_bulk_player_histories(player_ids)

        hr_rows: list[dict] = []
        for _, row in df.iterrows():
            pid = int(row["player_id"])
            hist = all_histories.get(pid, pd.DataFrame())
            opp_tid = (
                int(row["opp_team_id"])
                if pd.notna(row.get("opp_team_id"))
                else None
            )
            hr_rows.append(compute_hit_rates(hist, selected_stat, vegas_line, opp_tid))

        hr_df = pd.DataFrame(hr_rows, columns=_HIT_RATE_COLS)
        display = pd.concat([display.reset_index(drop=True), hr_df], axis=1)
        display.index = range(1, len(display) + 1)

        # pandas 3.x: Styler.applymap was renamed to Styler.map
        styled = (
            display.style
            .format(
                {"Predicted": lambda v: f"{v:.1f}" if v is not None and not pd.isna(v) else "—"},
            )
            .format({c: "{:.0f}%" for c in _HIT_RATE_COLS}, na_rep="—")
            .map(_pct_css, subset=_HIT_RATE_COLS)
        )
        st.dataframe(styled, use_container_width=True, height=min(800, 36 + 35 * len(display)))
    else:
        # No line entered — show full multi-stat prediction table
        base = df[
            ["full_name", "team", "opp", "home_away", "rest_days",
             "predicted_pts", "predicted_reb", "predicted_ast", "predicted_fg3m"]
        ].copy()
        base["pra"] = (
            df["predicted_pts"] + df["predicted_reb"] + df["predicted_ast"]
        )
        base.columns = [
            "Player", "Team", "Opp", "H/A", "Rest", "Pts", "Reb", "Ast", "3PM", "PRA"
        ]
        base = base.reset_index(drop=True)
        base.index = range(1, len(base) + 1)
        st.caption("Enter a Vegas line above to see Pickfinder-style hit-rate columns.")
        st.dataframe(
            base,
            use_container_width=True,
            column_config={
                "Pts": st.column_config.NumberColumn(format="%.1f"),
                "Reb": st.column_config.NumberColumn(format="%.1f"),
                "Ast": st.column_config.NumberColumn(format="%.1f"),
                "3PM": st.column_config.NumberColumn(format="%.1f"),
                "PRA": st.column_config.NumberColumn(format="%.1f"),
            },
            height=min(800, 36 + 35 * len(base)),
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
        st.caption(
            "No trained model found — run `uv run python -m nba_predictor.model.train` first."
        )


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

    sel_col, line_col, opp_col = st.columns([2, 1, 1])
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
            help="Enter a line to display Over/Under probability and hit rates",
        )
    with opp_col:
        selected_opp: str = st.selectbox(
            "H2H Opponent",
            options=list(_TEAM_OPTIONS.keys()),
            index=0,
            help="Filter the H2H hit-rate tile to games vs this opponent",
        )
    vegas_line: float | None = float(vegas_line_raw) if vegas_line_raw > 0 else None
    h2h_opp_tid: int | None = _TEAM_OPTIONS[selected_opp]
    stat_label = _STAT_LABELS[selected_stat]

    # Tonight's slate row for this player (None if not playing tonight)
    player_row: dict | None = None
    if not slate_df.empty:
        rows = slate_df[slate_df["player_id"] == selected_id]
        if not rows.empty:
            player_row = rows.iloc[0].to_dict()

    pred_val = _get_pred_val(player_row, selected_stat) if player_row else None

    st.markdown(f"### {player_options.get(selected_id, selected_id)}")

    if player_row:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            f"Predicted {stat_label}",
            f"{pred_val:.1f}" if pred_val is not None else "—",
        )
        col2.metric("Opponent", player_row["opp"])
        rest = player_row.get("rest_days")
        col3.metric("Rest Days", int(rest) if rest is not None else "—")
        col4.metric("Home/Away", player_row.get("home_away", "—"))
    else:
        st.info("Not on tonight's slate — showing recent history below.")

    # Over/Under probability (only when a Vegas line is entered)
    if vegas_line is not None and pred_val is not None:
        if selected_stat in _STAT_COMBOS:
            sub_params = [
                load_residual_params_cached(s) for s in _STAT_COMBOS[selected_stat]
            ]
            if all(sd is not None for _, sd in sub_params):
                r_mean: float | None = sum((m or 0.0) for m, _ in sub_params)
                r_std: float | None = float(
                    sum(sd ** 2 for _, sd in sub_params) ** 0.5
                )
            else:
                r_mean, r_std = None, None
        else:
            r_mean, r_std = load_residual_params_cached(selected_stat)

        if r_std is not None:
            probs = compute_probability(pred_val, vegas_line, r_std, r_mean or 0.0)
            p_over, p_under, edge = probs["p_over"], probs["p_under"], probs["edge"]

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
        else:
            st.caption(
                "Calibration unavailable — run retrain to enable probability estimates."
            )

    # History — required for hit-rate tiles and bar chart
    history = load_player_history(selected_id)
    if history.empty:
        st.warning("No game history found for this player.")
        return

    # Drop DNP rows (minutes == 0 or null) so they don't create chart gaps
    # or drag hit-rate percentages down with phantom 0-stat games.
    history = history[history["minutes"].fillna(0) > 0].reset_index(drop=True)

    # Pickfinder hit-rate tiles (only when a line is entered)
    if vegas_line is not None:
        rates = compute_hit_rates(history, selected_stat, vegas_line, h2h_opp_tid)
        st.markdown(
            _hit_rate_html(rates, vegas_line, stat_label),
            unsafe_allow_html=True,
        )

    # Window selector — matches the Pickfinder column tabs (L5 / L10 / L15 / L20 / L30)
    selected_window: str = st.radio(
        "Chart window",
        options=list(_WINDOW_SIZES.keys()),
        index=3,          # default to L20
        horizontal=True,
        help="Number of most-recent games shown (filtered to selected opponent if H2H is set)",
    )
    n_chart = _WINDOW_SIZES[selected_window]

    # When an H2H opponent is chosen, restrict the chart to games vs that team only.
    if h2h_opp_tid is not None and "opp_team_id" in history.columns:
        chart_pool = history[history["opp_team_id"] == h2h_opp_tid]
        chart_label = f" vs {selected_opp}"
    else:
        chart_pool = history
        chart_label = ""
    display_history = chart_pool.tail(n_chart)

    if display_history.empty:
        st.info(f"No recorded games vs {selected_opp} in the last 35 games.")
        return

    # Compute stat values for the chart window
    if selected_stat in ("pts", "reb", "ast", "fg3m"):
        bar_vals = display_history[selected_stat].fillna(0).to_numpy(dtype=float)
    else:
        bar_vals = sum(
            display_history[c].fillna(0) for c in _STAT_COMBOS[selected_stat]
        ).to_numpy(dtype=float)

    # Reference line priority: Vegas line > tonight's prediction > rolling average
    if vegas_line is not None:
        ref_line = vegas_line
        ref_label = f"Line: {ref_line:.1f}"
    elif pred_val is not None:
        ref_line = pred_val
        ref_label = f"Predicted: {ref_line:.1f}"
    else:
        ref_line = float(bar_vals.mean()) if len(bar_vals) >= 1 else 0.0
        ref_label = f"Avg: {ref_line:.1f}"

    bar_colors = ["#2ecc71" if v > ref_line else "#e74c3c" for v in bar_vals]
    y_max = (
        max(float(bar_vals.max()) * 1.3, ref_line * 1.3, 5.0)
        if len(bar_vals)
        else 10.0
    )

    # Convert dates to strings so Plotly uses a categorical (not time-series)
    # axis — this prevents gap-filling between non-consecutive game dates.
    x_labels = display_history["game_date"].dt.strftime("%b %-d").tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_labels,
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
        title=f"Last {n_chart} Games{chart_label} — {stat_label}  (green = over reference line)",
        xaxis_title="Date",
        xaxis={"type": "category"},
        yaxis_title=stat_label,
        yaxis={"range": [0, y_max]},
        height=380,
        margin={"t": 60, "b": 40, "l": 50, "r": 20},
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature snapshot bar chart
    opp_stats: dict = {}
    if player_row and player_row.get("opp_team_id"):
        opp_stats = load_opp_season_stats(int(player_row["opp_team_id"]))

    rest_days_val = (
        int(player_row["rest_days"])
        if player_row and player_row.get("rest_days") is not None
        else None
    )
    is_home_val = (
        int(player_row["is_home"])
        if player_row and player_row.get("is_home") is not None
        else None
    )

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
                return "#f0803c"
            if name in {
                "rest_days", "is_home", "is_back_to_back",
                "is_cold_start", "games_played_season",
            }:
                return "#2ecc71"
            return "#3498db"

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
            title=(
                f"Feature Snapshot — {stat_label}"
                "  (blue: form, green: context, orange: opponent)"
            ),
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

    with st.sidebar:
        st.markdown("### Data")
        if st.button("🔄 Refresh data", help="Clear cached DB queries and reload fresh data"):
            st.cache_data.clear()
            st.rerun()

    slate_df = load_tonights_slate()
    model_info = load_model_info()

    tab1, tab2 = st.tabs(["Tonight's Slate", "Player Lookup"])

    with tab1:
        _render_slate_tab(slate_df, model_info)

    with tab2:
        _render_player_tab(slate_df)


if __name__ == "__main__":
    main()
