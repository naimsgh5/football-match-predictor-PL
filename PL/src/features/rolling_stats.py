"""Recent form and goals scored/conceded, as a rolling average over the last n matches.

Strict pre-match computation: each value for a given match only uses that team's
earlier matches.
"""
import numpy as np
import pandas as pd

WINDOW = 10
CONGESTION_WINDOW_DAYS = 10


def add_form_features(df: pd.DataFrame, n: int = WINDOW):
    """Adds form_home, form_away, form_diff (average result over the last n matches,
    1=win, 0.5=draw, 0=loss). Also returns the final per-team history."""
    history: dict[str, list[float]] = {}
    form_home, form_away = [], []

    for home, away, hs, aw in zip(df["home_team"], df["away_team"], df["home_score"], df["away_score"]):
        h_hist = history.get(home, [])
        a_hist = history.get(away, [])
        form_home.append(np.mean(h_hist[-n:]) if h_hist else 0.5)
        form_away.append(np.mean(a_hist[-n:]) if a_hist else 0.5)

        if hs > aw:
            home_result, away_result = 1.0, 0.0
        elif hs == aw:
            home_result, away_result = 0.5, 0.5
        else:
            home_result, away_result = 0.0, 1.0
        history.setdefault(home, []).append(home_result)
        history.setdefault(away, []).append(away_result)

    out = df.copy()
    out["form_home"] = form_home
    out["form_away"] = form_away
    out["form_diff"] = out["form_home"] - out["form_away"]
    return out, history


def add_venue_form_features(df: pd.DataFrame, n: int = WINDOW):
    """Venue-specific form: unlike form_diff (which mixes each team's home+away matches),
    here the home team is evaluated only on its last n HOME matches, and the away team only
    on its last n AWAY matches — some teams are much stronger at home than on the road."""
    home_history: dict[str, list[float]] = {}
    away_history: dict[str, list[float]] = {}
    form_home_specific, form_away_specific = [], []

    for home, away, hs, aw in zip(df["home_team"], df["away_team"], df["home_score"], df["away_score"]):
        h_hist = home_history.get(home, [])
        a_hist = away_history.get(away, [])
        form_home_specific.append(np.mean(h_hist[-n:]) if h_hist else 0.5)
        form_away_specific.append(np.mean(a_hist[-n:]) if a_hist else 0.5)

        if hs > aw:
            home_result, away_result = 1.0, 0.0
        elif hs == aw:
            home_result, away_result = 0.5, 0.5
        else:
            home_result, away_result = 0.0, 1.0
        home_history.setdefault(home, []).append(home_result)
        away_history.setdefault(away, []).append(away_result)

    out = df.copy()
    out["form_home_specific"] = form_home_specific
    out["form_away_specific"] = form_away_specific
    out["venue_form_diff"] = out["form_home_specific"] - out["form_away_specific"]
    return out, home_history, away_history


def add_congestion_features(df: pd.DataFrame, window_days: int = CONGESTION_WINDOW_DAYS):
    """Proxy for fixture congestion (and thus squad-rotation risk): number of matches
    played by each team in the `window_days` days preceding the current match (excluding
    it). Known limitation: only counts Premier League matches present in this dataset, not
    cup/European matches (not available in the source) — so it underestimates the true
    fatigue of a team competing on multiple fronts, but stays computable without a new
    data source.

    congestion_diff = congestion_away - congestion_home (same convention as the other
    _diff columns: positive = favors the home team, here because the away team has played
    more recently/often)."""
    match_dates: dict[str, list[pd.Timestamp]] = {}
    congestion_home, congestion_away = [], []

    for date, home, away in zip(df["date"], df["home_team"], df["away_team"]):
        cutoff = date - pd.Timedelta(days=window_days)
        congestion_home.append(sum(1 for d in match_dates.get(home, []) if d > cutoff))
        congestion_away.append(sum(1 for d in match_dates.get(away, []) if d > cutoff))

        match_dates.setdefault(home, []).append(date)
        match_dates.setdefault(away, []).append(date)

    out = df.copy()
    out["congestion_home"] = congestion_home
    out["congestion_away"] = congestion_away
    out["congestion_diff"] = out["congestion_away"] - out["congestion_home"]
    return out, match_dates


def add_goals_features(df: pd.DataFrame, n: int = WINDOW, default_avg: float = 1.3):
    """Adds rolling averages of goals scored/conceded and the attack/defense diffs.

    The fallback for a team with no history (its very first match in the dataset) uses
    the cumulative goals average of matches already processed (not the whole dataset's
    average, which would use future matches) — before the very first match ever, a fixed
    default_avg (~typical PL average) seeds it.
    """
    scored: dict[str, list[int]] = {}
    conceded: dict[str, list[int]] = {}
    gs_home, gc_home, gs_away, gc_away = [], [], [], []

    running_total_goals = 0.0
    running_count = 0

    def avg(team, store, fallback):
        hist = store.get(team, [])
        return np.mean(hist[-n:]) if hist else fallback

    for home, away, hs, aw in zip(df["home_team"], df["away_team"], df["home_score"], df["away_score"]):
        current_avg = (running_total_goals / running_count) if running_count else default_avg

        gs_home.append(avg(home, scored, current_avg))
        gc_home.append(avg(home, conceded, current_avg))
        gs_away.append(avg(away, scored, current_avg))
        gc_away.append(avg(away, conceded, current_avg))

        scored.setdefault(home, []).append(hs)
        conceded.setdefault(home, []).append(aw)
        scored.setdefault(away, []).append(aw)
        conceded.setdefault(away, []).append(hs)

        running_total_goals += hs + aw
        running_count += 2

    out = df.copy()
    out["goals_scored_home"] = gs_home
    out["goals_conceded_home"] = gc_home
    out["goals_scored_away"] = gs_away
    out["goals_conceded_away"] = gc_away
    out["attack_diff"] = out["goals_scored_home"] - out["goals_scored_away"]
    out["defense_diff"] = out["goals_conceded_away"] - out["goals_conceded_home"]
    return out, scored, conceded
