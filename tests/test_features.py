"""M2 milestone: feature leak-proof validation.

Three structural invariants that must hold for every row in the feature
DataFrame, regardless of player, season, or date range:

  1. First-game NaN — every player's first game has NaN for all rolling and
     season-average features, because shift(1) produces NaN for the first
     element of each player group.

  2. Second-game correctness — roll10_pts for game 2 equals the pts from
     game 1 (the only prior game, so min_periods=1 collapses to that one
     value). This confirms the rolling window is reading the shifted lag,
     not the current row.

  3. Season reset — season_avg_pts is NaN for a player's first game of a
     new season (within-season shift(1) resets), while roll10_pts is NOT
     NaN for the same row (cross-season lag carries over prior-season form).
     This confirms the two-lag design is working correctly.

These tests require a live DB connection. They are skipped when
SUPABASE_DB_URL is absent so the suite stays runnable offline.
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("SUPABASE_DB_URL"),
    reason="SUPABASE_DB_URL not set — skipping DB integration tests",
)

# Mid-season snapshot: enough history to cover multi-season players and all
# cold-start cases, small enough that the build is fast in CI (~10 s).
_AS_OF = date(2023, 1, 31)


@pytest.fixture(scope="module")
def features():
    from nba_predictor.features.build import build_features

    return build_features(as_of_date=_AS_OF)


# --------------------------------------------------------------------------- tests


def test_first_game_rolling_features_are_nan(features):
    """Every player's first-ever game must have NaN rolling features.

    shift(1) produces NaN for the first row of each player group. A rolling
    window on all-NaN input returns NaN regardless of min_periods.

    Uses drop_duplicates(keep="first") rather than groupby().first() because
    pandas groupby.first() returns the first *non-NaN* value, not the first row.
    """
    first = features.sort_values("game_date").drop_duplicates(subset=["player_id"], keep="first")
    for col in ("roll5_pts", "roll10_pts", "season_avg_pts"):
        nulls = first[col].isna()
        assert nulls.all(), (
            f"{col}: {(~nulls).sum()} players have non-NaN on their first game — "
            "shift(1) may not be applied"
        )


def test_second_game_roll10_equals_first_game_pts(features):
    """For game 2, roll10_pts must equal game 1's pts (only one prior game).

    With min_periods=1, rolling(10).mean() over a single non-NaN value
    returns that value exactly. Any deviation means the rolling window is
    including the current game (off-by-one shift error).
    """
    df = features.sort_values(["player_id", "game_date"]).copy()
    df["_game_num"] = df.groupby("player_id").cumcount() + 1

    first = df[df["_game_num"] == 1][["player_id", "pts"]].rename(columns={"pts": "game1_pts"})
    second = df[df["_game_num"] == 2][["player_id", "roll10_pts"]].dropna(subset=["roll10_pts"])
    joined = second.merge(first, on="player_id", how="inner")

    assert len(joined) > 0, "No players with ≥ 2 games found — check as_of date"

    mismatched = ~np.isclose(joined["roll10_pts"], joined["game1_pts"], atol=1e-3)
    assert not mismatched.any(), (
        f"{mismatched.sum()} players have roll10_pts ≠ game1_pts on game 2:\n"
        f"{joined[mismatched][['player_id', 'roll10_pts', 'game1_pts']].head()}"
    )


def test_season_avg_resets_at_new_season(features):
    """season_avg_pts is NaN at game 1 of a new season; roll10_pts is not.

    The within-season lag (_s_pts) resets per (player, season), so game 1
    of any season has no prior-season data in the expanding average.
    The cross-season lag (_x_pts) does NOT reset, so roll10_pts carries
    prior-season form into the new season's first game.

    This test confirms both lag variants are wired to the right features.
    """
    # Find players with games in both 2021-22 and 2022-23
    multi = features.groupby("player_id")["season"].nunique()
    multi_players = multi[multi >= 2].index
    assert len(multi_players) > 0, "No multi-season players found — check as_of date"

    sub = features[features["player_id"].isin(multi_players)]
    first_of_2223 = (
        sub[sub["season"] == "2022-23"]
        .sort_values("game_date")
        .drop_duplicates(subset=["player_id"], keep="first")
    )

    # season_avg_pts must be NaN: within-season lag has no data yet
    season_avg_not_null = first_of_2223["season_avg_pts"].notna()
    assert not season_avg_not_null.any(), (
        f"{season_avg_not_null.sum()} players have non-NaN season_avg_pts at "
        "start of 2022-23 — within-season lag is not resetting correctly"
    )

    # roll10_pts must NOT be all NaN: cross-season lag carries 2021-22 form
    roll10_all_null = first_of_2223["roll10_pts"].isna().all()
    assert not roll10_all_null, (
        "roll10_pts is NaN for all multi-season players at start of 2022-23 — "
        "cross-season lag is not carrying prior-season history"
    )


def test_games_played_season_is_zero_at_game_1(features):
    """games_played_season must be 0 for every player's first game of each season."""
    first_per_season = features.sort_values("game_date").drop_duplicates(
        subset=["player_id", "season"], keep="first"
    )
    bad = first_per_season[first_per_season["games_played_season"] != 0]
    assert bad.empty, (
        f"{len(bad)} (player, season) pairs have games_played_season != 0 at "
        "their season's first game"
    )


def test_no_pts_self_inclusion_in_season_avg(features):
    """season_avg_pts must be NaN on game 1 of every (player, season).

    If the within-season shift were missing, season_avg_pts for game 1 would
    equal that game's pts (expanding mean over a single value = that value).
    With correct shift(1), season_avg_pts for game 1 is NaN.
    """
    first_per_season = features.sort_values("game_date").drop_duplicates(
        subset=["player_id", "season"], keep="first"
    )
    # Any non-NaN season_avg_pts on game 1 is a leak
    leaked = first_per_season["season_avg_pts"].notna()
    assert not leaked.any(), (
        f"{leaked.sum()} (player, season) first games have non-NaN season_avg_pts — "
        "the within-season shift(1) may be missing"
    )
