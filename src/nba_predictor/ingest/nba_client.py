"""Thin, well-behaved wrapper around `nba_api` endpoints.

Why a wrapper instead of calling nba_api directly from each script:
  * **Rate limiting.** NBA.com throttles aggressive scrapers; ~1 req/sec is
    the de-facto community standard. We centralize a tiny throttle here so
    every caller is automatically polite.
  * **Retries.** The endpoints are flaky (occasional 5xx, ReadTimeout). We
    wrap each call in tenacity exponential backoff so transient failures
    don't kill a multi-hour backfill.
  * **Typed surface.** Each method returns a `pandas.DataFrame` with a
    documented row shape. Callers don't need to know about nba_api's
    `get_data_frames()[0]` idiom.
  * **One place to mock.** Tests stub `NBAClient.get_*` methods rather than
    monkey-patching nba_api internals.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pandas as pd
from nba_api.stats.endpoints import (
    commonallplayers,
    leaguedashteamstats,
    playergamelogs,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# nba_api raises requests exceptions; we retry on transient network errors
# but NOT on stuff like KeyError (which would mean a schema change we need
# to surface, not silently retry).
_RETRYABLE_EXC = (TimeoutError, ConnectionError, OSError)


@dataclass
class NBAClient:
    """Polite, retrying client for the subset of nba_api endpoints we need.

    Attributes
    ----------
    request_timeout : float
        Per-request HTTP timeout passed through to nba_api.
    min_interval : float
        Minimum seconds between successive requests. Defaults to 0.6s,
        leaving headroom under the ~1 req/sec community guideline so bursty
        retries don't trip rate limiting.
    """

    request_timeout: float = 60.0
    min_interval: float = 0.6
    _last_call: float = 0.0

    # ------------------------------------------------------------------ utils

    def _throttle(self) -> None:
        """Sleep just long enough to respect `min_interval` since last call."""
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()

    # ------------------------------------------------------------------ endpoints

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_EXC),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def get_all_players(self, *, only_current_season: bool = False) -> pd.DataFrame:
        """Return one row per player from `CommonAllPlayers`.

        Columns include `PERSON_ID`, `DISPLAY_FIRST_LAST`, `ROSTERSTATUS`,
        `TEAM_ID`, `FROM_YEAR`, `TO_YEAR`. `ROSTERSTATUS == 1` means active.
        """
        self._throttle()
        endpoint = commonallplayers.CommonAllPlayers(
            is_only_current_season=int(only_current_season),
            timeout=self.request_timeout,
        )
        return endpoint.get_data_frames()[0]

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_EXC),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def get_player_game_logs(self, season: str) -> pd.DataFrame:
        """All player game logs for a given season (e.g. ``"2024-25"``).

        One row per (player, game). Columns include `PLAYER_ID`, `GAME_ID`,
        `GAME_DATE`, `TEAM_ID`, `MIN`, `PTS`, `REB`, `AST`, `FGM`, `FGA`,
        `FG3M`, `FG3A`, `FTM`, `FTA`, `STL`, `BLK`, `TOV`, `PLUS_MINUS`, and
        a `MATCHUP` string we parse for home/away.

        We pull the entire season in one call (the endpoint pages internally),
        which is far gentler on the API than per-player iteration.
        """
        self._throttle()
        endpoint = playergamelogs.PlayerGameLogs(
            season_nullable=season,
            timeout=self.request_timeout,
        )
        return endpoint.get_data_frames()[0]

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_EXC),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def get_team_advanced_per_game(self, season: str) -> pd.DataFrame:
        """Season-level advanced team stats (pace, off/def rating).

        Used in Phase 1.4 as a per-team baseline; we'll later refine to
        per-game advanced metrics if rolling features need finer granularity.
        Columns include `TEAM_ID`, `OFF_RATING`, `DEF_RATING`, `PACE`.
        """
        self._throttle()
        endpoint = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            timeout=self.request_timeout,
        )
        return endpoint.get_data_frames()[0]


# Singleton accessor — we want every caller in a process to share one
# throttle clock, otherwise concurrent ingest jobs each think they're under
# the limit and collectively blow past it.
_default_client: NBAClient | None = None


def get_client() -> NBAClient:
    global _default_client
    if _default_client is None:
        _default_client = NBAClient()
    return _default_client


if __name__ == "__main__":
    # Smoke test: `uv run python -m nba_predictor.ingest.nba_client`
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    client = get_client()

    logger.info("Fetching CommonAllPlayers (current season only)...")
    players = client.get_all_players(only_current_season=True)
    logger.info("Got %d players. Columns: %s", len(players), list(players.columns))

    logger.info("Fetching PlayerGameLogs for 2024-25...")
    logs = client.get_player_game_logs("2024-25")
    logger.info("Got %d rows. Sample columns: %s", len(logs), list(logs.columns)[:12])
