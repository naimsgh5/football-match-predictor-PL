"""Recent form and goals scored/conceded, as a rolling average over the last n matches.

Strict pre-match computation: each value for a given match only uses that team's
earlier matches.
"""
import numpy as np
import pandas as pd

WINDOW = 10
CONGESTION_WINDOW_DAYS = 10
CLEAN_SHEET_DEFAULT_RATE = 0.25  # ~league-wide clean sheet rate, seeds a team with no history yet


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


def _expected_result(own_elo: float, opponent_elo: float) -> float:
    """Same logistic formula as elo.py: probability of winning given the Elo gap."""
    return 1 / (1 + 10 ** ((opponent_elo - own_elo) / 400))


def add_quality_form_features(df: pd.DataFrame, n: int = WINDOW):
    """Form weighted by opponent strength: rolling average of (result - expected result),
    where the expected result comes from the same Elo formula as elo.py (df must already
    have elo_home/elo_away -- call this after add_elo_features).

    Unlike form_diff (which treats every win/draw/loss the same regardless of opponent),
    this rewards overperforming a tough opponent and penalizes underperforming a weak one:
    a draw against a much stronger team scores strongly positive, a loss to a much weaker
    team scores strongly negative, whereas form_diff would count them identically to a
    draw/loss against any other opponent. Answers "beating the 16th isn't the same as
    beating the 2nd" directly, on top of what elo_diff already partially captures.

    Not bounded to [0, 1] like form_diff (a rolling average of a signed, unbounded-ish
    quantity) -- fine since it goes through StandardScaler before training anyway."""
    history: dict[str, list[float]] = {}
    qform_home, qform_away = [], []

    for home, away, hs, aw, elo_home, elo_away in zip(
        df["home_team"], df["away_team"], df["home_score"], df["away_score"], df["elo_home"], df["elo_away"]
    ):
        h_hist = history.get(home, [])
        a_hist = history.get(away, [])
        qform_home.append(np.mean(h_hist[-n:]) if h_hist else 0.0)
        qform_away.append(np.mean(a_hist[-n:]) if a_hist else 0.0)

        if hs > aw:
            home_result, away_result = 1.0, 0.0
        elif hs == aw:
            home_result, away_result = 0.5, 0.5
        else:
            home_result, away_result = 0.0, 1.0

        exp_home = _expected_result(elo_home, elo_away)
        exp_away = 1 - exp_home
        history.setdefault(home, []).append(home_result - exp_home)
        history.setdefault(away, []).append(away_result - exp_away)

    out = df.copy()
    out["quality_form_home"] = qform_home
    out["quality_form_away"] = qform_away
    out["quality_form_diff"] = out["quality_form_home"] - out["quality_form_away"]
    return out, history


def add_clean_sheet_features(df: pd.DataFrame, n: int = WINDOW, default_rate: float = CLEAN_SHEET_DEFAULT_RATE):
    """Rolling clean sheet rate (fraction of the last n matches with 0 goals conceded).

    Distinct from defense_diff (average goals conceded): two teams can have the same
    average while having very different defensive consistency, e.g. conceding
    [0,0,0,0,3] vs [1,1,1,1,1] both average 0.6 goals/match but the first has 4 clean
    sheets out of 5 and the second has none -- defense_diff alone can't tell them apart.

    default_rate seeds a team with no history yet (~typical league-wide clean sheet rate,
    refine per league if the actual rate differs noticeably)."""
    clean_sheets: dict[str, list[float]] = {}
    cs_home, cs_away = [], []

    for home, away, hs, aw in zip(df["home_team"], df["away_team"], df["home_score"], df["away_score"]):
        h_hist = clean_sheets.get(home, [])
        a_hist = clean_sheets.get(away, [])
        cs_home.append(np.mean(h_hist[-n:]) if h_hist else default_rate)
        cs_away.append(np.mean(a_hist[-n:]) if a_hist else default_rate)

        home_clean = 1.0 if aw == 0 else 0.0
        away_clean = 1.0 if hs == 0 else 0.0
        clean_sheets.setdefault(home, []).append(home_clean)
        clean_sheets.setdefault(away, []).append(away_clean)

    out = df.copy()
    out["clean_sheet_rate_home"] = cs_home
    out["clean_sheet_rate_away"] = cs_away
    out["clean_sheet_diff"] = out["clean_sheet_rate_home"] - out["clean_sheet_rate_away"]
    return out, clean_sheets


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


def add_venue_goals_features(df: pd.DataFrame, n: int = WINDOW, default_avg: float = 1.3):
    """Rolling average of goals scored/conceded, split by venue: the home team's numbers
    only come from ITS past HOME matches, the away team's only from ITS past AWAY matches
    -- unlike attack_diff/defense_diff (add_goals_features above), which pool a team's home
    and away matches together regardless of where the upcoming match is played.

    Used by the Poisson goal model (src/models/markets.py) so lambda_home/lambda_away
    reflect each team's own measured home/away scoring split directly, instead of a flat
    league-wide HOME_ADVANTAGE multiplier applied on top of venue-mixed averages. NOT a
    training feature (not added to FEATURE_COLUMNS) -- the two classifiers (LR/MLP) keep
    learning from the venue-mixed attack_diff/defense_diff they were trained on; changing
    that would mean retraining them, which is a separate decision from improving the
    (untrained, purely descriptive) goal model."""
    home_scored: dict[str, list[int]] = {}
    home_conceded: dict[str, list[int]] = {}
    away_scored: dict[str, list[int]] = {}
    away_conceded: dict[str, list[int]] = {}
    vgs_home, vgc_home, vgs_away, vgc_away = [], [], [], []

    for home, away, hs, aw in zip(df["home_team"], df["away_team"], df["home_score"], df["away_score"]):
        h_gs = home_scored.get(home, [])
        h_gc = home_conceded.get(home, [])
        a_gs = away_scored.get(away, [])
        a_gc = away_conceded.get(away, [])
        vgs_home.append(np.mean(h_gs[-n:]) if h_gs else default_avg)
        vgc_home.append(np.mean(h_gc[-n:]) if h_gc else default_avg)
        vgs_away.append(np.mean(a_gs[-n:]) if a_gs else default_avg)
        vgc_away.append(np.mean(a_gc[-n:]) if a_gc else default_avg)

        home_scored.setdefault(home, []).append(hs)
        home_conceded.setdefault(home, []).append(aw)
        away_scored.setdefault(away, []).append(aw)
        away_conceded.setdefault(away, []).append(hs)

    out = df.copy()
    out["venue_goals_scored_home"] = vgs_home
    out["venue_goals_conceded_home"] = vgc_home
    out["venue_goals_scored_away"] = vgs_away
    out["venue_goals_conceded_away"] = vgc_away
    return out, home_scored, home_conceded, away_scored, away_conceded


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
