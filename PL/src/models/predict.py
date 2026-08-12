"""Predicts an upcoming match (never seen during training) with the already-trained model.

Recomputes the "core" features (Elo, form, home/away form, h2h, goals, 5-season rank)
from the final state of the full history, then applies optional *post-hoc* adjustments
(the model never learned from these, they only shift the final probability, as in
algo/CLUBS_LOGISTIC_REGRESSION.ipynb): injuries/market value, rest days, current
standings/points, bookmaker odds.

Usage:
    python -m src.models.predict "Arsenal" "Chelsea"
"""
import sys

import joblib
import numpy as np
import pandas as pd

from src.evaluation.metrics import RESULT_LABELS
from src.features.build_dataset import FEATURE_COLUMNS, build_dataset_with_state
from src.features.head_to_head import WINDOW as H2H_WINDOW
from src.features.historical_rank import N_SEASONS, average_rank
from src.features.market_value_injuries import injury_strength
from src.features.rolling_stats import CONGESTION_WINDOW_DAYS, WINDOW as FORM_WINDOW
from src.features.squad_values import SQUAD_VALUES
from src.models.markets import OU_LINES, betting_notes, btts, expected_goals, goal_matrix, implied_1x2, over_under, top_scorelines

MODEL_PATH = "models_saved/baseline_lr.joblib"
SCALER_PATH = "models_saved/baseline_lr_scaler.joblib"
DEFAULT_GOALS_AVG = 1.3
REST_DAY_IMPACT = 0.12       # maximum (asymptotic) impact of rest days on the probabilities
INJURY_IMPACT = 0.15         # maximum impact of the injury factor on the probabilities
STANDINGS_IMPACT = 0.15      # maximum impact of the current points gap on the probabilities
STANDINGS_POINTS_SCALE = 15  # points gap (~15 pts) beyond which the impact saturates
ODDS_BLEND_WEIGHT = 0.5      # 0 = pure model, 1 = pure odds


# ---------------------------------------------------------------------------
# "Core" features (same computations as at training time, applied outside the dataset)
# ---------------------------------------------------------------------------

def _team_form(team, form_history, n=FORM_WINDOW):
    hist = form_history.get(team, [])
    return np.mean(hist[-n:]) if hist else 0.5


def _venue_form(team, venue_history, n=FORM_WINDOW):
    hist = venue_history.get(team, [])
    return np.mean(hist[-n:]) if hist else 0.5


def _team_goals_avg(team, store, n=FORM_WINDOW):
    hist = store.get(team, [])
    return np.mean(hist[-n:]) if hist else DEFAULT_GOALS_AVG


def _h2h_rate(home, away, h2h_history, n=H2H_WINDOW):
    key = tuple(sorted([home, away]))
    ref_team = key[0]
    hist = h2h_history.get(key, [])
    avg_ref = np.mean(hist[-n:]) if hist else 0.5
    return avg_ref if home == ref_team else 1 - avg_ref


def _rank_diff(home, away, standings, n_seasons=N_SEASONS):
    seasons_sorted = sorted(standings.keys())[-n_seasons:]
    rank_home = average_rank(home, standings, seasons_sorted)
    rank_away = average_rank(away, standings, seasons_sorted)
    return rank_away - rank_home, rank_home, rank_away


def _ordinal(n: int) -> str:
    """1 -> '1st', 2 -> '2nd', 3 -> '3rd', 11-13 -> 'th', 21 -> '21st', etc."""
    if 11 <= n % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _congestion(team, match_dates, target_date, window_days=CONGESTION_WINDOW_DAYS):
    """Number of PL matches played by `team` in the `window_days` days before `target_date`.
    Same limitation as at training time: only sees PL matches from this dataset, not
    cup/European matches."""
    cutoff = target_date - pd.Timedelta(days=window_days)
    dates = match_dates.get(team, [])
    return sum(1 for d in dates if cutoff < d < target_date)


def compute_features(home: str, away: str, state: dict, match_date=None):
    elo_home = state["elo"].get(home, 1500)
    elo_away = state["elo"].get(away, 1500)

    form_diff = _team_form(home, state["form_history"]) - _team_form(away, state["form_history"])
    venue_form_diff = (
        _venue_form(home, state["home_venue_history"]) - _venue_form(away, state["away_venue_history"])
    )
    h2h_rate = _h2h_rate(home, away, state["h2h_history"])

    gs_home = _team_goals_avg(home, state["scored"])
    gc_home = _team_goals_avg(home, state["conceded"])
    gs_away = _team_goals_avg(away, state["scored"])
    gc_away = _team_goals_avg(away, state["conceded"])

    rank_diff, rank_home, rank_away = _rank_diff(home, away, state["standings"])

    target_date = pd.Timestamp(match_date) if match_date is not None else pd.Timestamp.now().normalize()
    congestion_home = _congestion(home, state["match_dates"], target_date)
    congestion_away = _congestion(away, state["match_dates"], target_date)

    features = {
        "elo_diff": elo_home - elo_away,
        "form_diff": form_diff,
        "venue_form_diff": venue_form_diff,
        "h2h_home_win_rate": h2h_rate,
        "attack_diff": gs_home - gs_away,
        "defense_diff": gc_away - gc_home,
        "rank_diff": rank_diff,
        "congestion_diff": congestion_away - congestion_home,
    }
    context = {
        "elo_home": elo_home, "elo_away": elo_away,
        "rank_home": rank_home, "rank_away": rank_away,
        "congestion_home": congestion_home, "congestion_away": congestion_away,
        "gs_home": gs_home, "gc_home": gc_home, "gs_away": gs_away, "gc_away": gc_away,
    }
    return features, context


# ---------------------------------------------------------------------------
# Post-hoc adjustments (entered by hand, the model doesn't know about them)
# ---------------------------------------------------------------------------

def _apply_rest_days(p_home, p_draw, p_away, rest_days_diff):
    """rest_days_diff: home rest days - away rest days.
    +2 = home has 2 more rest days, -2 = the away team is more rested."""
    factor = float(np.tanh(rest_days_diff / 7.0) * REST_DAY_IMPACT)
    p_home = max(0.01, p_home + factor * 0.5 * (1 - p_home))
    p_away = max(0.01, p_away - factor * 0.5 * p_away)
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total, factor


def _apply_injuries(home, away, p_home, p_draw, p_away, injured_home, injured_away, squad_values):
    str_home, missing_home, details_home, _ = injury_strength(home, injured_home or [], squad_values)
    str_away, missing_away, details_away, _ = injury_strength(away, injured_away or [], squad_values)
    delta = (str_away - str_home) * INJURY_IMPACT
    p_home = max(0.01, p_home + delta)
    p_away = max(0.01, p_away - delta)
    total = p_home + p_draw + p_away
    info = {"home": details_home, "away": details_away}
    return p_home / total, p_draw / total, p_away / total, info


def _apply_current_standings(p_home, p_draw, p_away, home_points, away_points):
    """home_points/away_points: points in the current standings (current season). The
    points gap pushes the probability towards the higher-ranked team - saturates beyond
    ~15 points of gap (STANDINGS_POINTS_SCALE)."""
    points_diff = home_points - away_points
    factor = float(np.tanh(points_diff / STANDINGS_POINTS_SCALE) * STANDINGS_IMPACT)
    p_home = max(0.01, p_home + factor * 0.5 * (1 - p_home))
    p_away = max(0.01, p_away - factor * 0.5 * p_away)
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total, factor


def _print_goal_markets(home, away, context, p_home, p_draw, p_away, n_scorelines=5):
    """Prints probable scores / BTTS / over-under, via the Poisson goal model from
    src/models/markets.py -- INDEPENDENT of the main 1X2 model (see markets.py docstring)."""
    lambda_home, lambda_away = expected_goals(
        context["gs_home"], context["gc_home"], context["gs_away"], context["gc_away"]
    )
    matrix = goal_matrix(lambda_home, lambda_away)

    print(f"\n  --- Derived markets (Poisson goal model, independent of the 1X2 model above) ---")
    print(f"  Expected goals: {home} {lambda_home:.2f} - {lambda_away:.2f} {away}")

    imp = implied_1x2(matrix)
    print(f"  1X2 implied by this model: Home {imp['home']*100:.1f}% / Draw {imp['draw']*100:.1f}% / "
          f"Away {imp['away']*100:.1f}% (to compare, not confuse, with the 1X2 above)")

    print(f"  Most probable scores:")
    for i, j, p in top_scorelines(matrix, n=n_scorelines):
        print(f"    {home} {i} - {j} {away}  ({p*100:.1f}%)")

    btts_probs = btts(matrix)
    print(f"  BTTS (both teams score) : Yes {btts_probs['yes']*100:.1f}% / No {btts_probs['no']*100:.1f}%")

    ou_probs = {}
    for line in OU_LINES:
        ou_probs[line] = over_under(matrix, line)
        print(f"  Over/Under {line} : Over {ou_probs[line]['over']*100:.1f}% / Under {ou_probs[line]['under']*100:.1f}%")

    print(f"\n  Notes (informational, not financial advice):")
    for note in betting_notes(p_home, p_draw, p_away, btts_probs, ou_probs):
        print(f"    - {note}")


def _remove_margin(odds_1x2: dict) -> dict:
    raw = {k: 1 / v for k, v in odds_1x2.items()}
    s = sum(raw.values())
    return {k: v / s for k, v in raw.items()}


def _apply_odds(p_home, p_draw, p_away, odds_1x2, blend=ODDS_BLEND_WEIGHT):
    implied = _remove_margin(odds_1x2)
    p_home = (1 - blend) * p_home + blend * implied["1"]
    p_draw = (1 - blend) * p_draw + blend * implied["X"]
    p_away = (1 - blend) * p_away + blend * implied["2"]
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total, implied


# Match stakes: categorical level -> motivation boost for the team in question.
# "derby" is separate (not a per-team level): a derby doesn't favor either side, it
# statistically makes the match tighter (more draws / narrow scorelines), so it boosts p_draw.
STAKES_LEVELS = {
    "neutral": 0.00,
    "europe": 0.04,    # Champions/Europa League qualification at stake
    "survival": 0.05,  # relegation battle: very motivating (fear of relegation)
    "title": 0.06,     # title race
}
DERBY_DRAW_BOOST = 0.05


def _apply_stakes(p_home, p_draw, p_away, stakes_home, stakes_away, derby):
    boost_home = STAKES_LEVELS.get(stakes_home, 0.0) if stakes_home else 0.0
    boost_away = STAKES_LEVELS.get(stakes_away, 0.0) if stakes_away else 0.0
    delta = boost_home - boost_away
    p_home = max(0.01, p_home + delta)
    p_away = max(0.01, p_away - delta)

    if derby:
        take_home = p_home * DERBY_DRAW_BOOST
        take_away = p_away * DERBY_DRAW_BOOST
        p_home -= take_home
        p_away -= take_away
        p_draw += take_home + take_away

    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_match(home: str, away: str, model=None, scaler=None, state=None,
                   injured_home=None, injured_away=None, squad_values=None,
                   rest_days_diff: int = None,
                   home_position: int = None, home_points: int = None,
                   away_position: int = None, away_points: int = None,
                   odds_1x2=None, odds_blend=ODDS_BLEND_WEIGHT,
                   stakes_home: str = None, stakes_away: str = None, derby: bool = False,
                   match_date=None, show_markets: bool = True):
    """Returns {"home": p, "draw": p, "away": p} and prints a summary.

    Computed automatically (no input needed): elo, form, home/away form, h2h, goals,
    5-season average rank, fixture congestion (congestion_diff, on match_date - defaults
    to today; only counts PL matches from this dataset, not cup/European matches --
    underestimates the true fatigue of a team competing on multiple fronts, complement
    with rest_days_diff if you know the real situation better).

    To enter yourself (the model doesn't know about these):
      - injured_home / injured_away: lists of absent player names
      - rest_days_diff: home rest days - away rest days (+2 = home more rested)
      - home_position/home_points, away_position/away_points: CURRENT standings for the
        ongoing season (absent from the historical dataset until the season is over)
      - odds_1x2: optional dict {"1": home_odds, "X": draw_odds, "2": away_odds}
      - stakes_home/stakes_away: "title" / "europe" / "survival" / "neutral" (default)
      - derby: True/False -- boosts the draw probability (tight match), doesn't favor either side

    show_markets (default True): also shows probable scores / BTTS / over-under via a
    second model (Poisson on goals, independent of the 1X2 -- see src/models/markets.py).
    """
    if state is None:
        _df, state = build_dataset_with_state()
    if model is None:
        model = joblib.load(MODEL_PATH)
    if scaler is None:
        scaler = joblib.load(SCALER_PATH)
    if squad_values is None:
        squad_values = SQUAD_VALUES

    for team in (home, away):
        if team not in state["elo"]:
            print(f"! '{team}' not found in the dataset (team never seen, or name misspelled).")

    features, context = compute_features(home, away, state, match_date=match_date)
    x = np.array([[features[c] for c in FEATURE_COLUMNS]])
    x_scaled = scaler.transform(x)
    proba = dict(zip(model.classes_, model.predict_proba(x_scaled)[0]))
    p_away, p_draw, p_home = proba.get(0, 0.0), proba.get(1, 0.0), proba.get(2, 0.0)

    print(f"\n{home} (home) vs {away} (away)")
    print(f"  Elo  : {context['elo_home']:.0f} vs {context['elo_away']:.0f}")
    print(f"  Average rank over last {N_SEASONS} seasons : {context['rank_home']:.1f} vs {context['rank_away']:.1f}")
    print(f"  Fixture congestion ({CONGESTION_WINDOW_DAYS}d, PL matches only) : "
          f"{home} {context['congestion_home']} vs {away} {context['congestion_away']}")
    print(f"  Model only -> Home {p_home*100:.1f}% / Draw {p_draw*100:.1f}% / Away {p_away*100:.1f}%")

    if injured_home or injured_away:
        p_home, p_draw, p_away, inj_info = _apply_injuries(
            home, away, p_home, p_draw, p_away, injured_home, injured_away, squad_values
        )
        print(f"  + Injuries : {home} missing {list(inj_info['home'])}, {away} missing {list(inj_info['away'])}")

    if rest_days_diff:
        p_home, p_draw, p_away, rest_factor = _apply_rest_days(p_home, p_draw, p_away, rest_days_diff)
        print(f"  + Rest : {home} {'+' if rest_days_diff >= 0 else ''}{rest_days_diff}d vs {away} "
              f"(impact {rest_factor*100:+.1f}%)")

    if home_points is not None and away_points is not None:
        p_home, p_draw, p_away, standings_factor = _apply_current_standings(p_home, p_draw, p_away, home_points, away_points)
        pos_str = (f" ({_ordinal(home_position)} vs {_ordinal(away_position)})" if home_position and away_position else "")
        print(f"  + Current standings : {home_points}pts vs {away_points}pts{pos_str} (impact {standings_factor*100:+.1f}%)")

    if stakes_home or stakes_away or derby:
        p_home, p_draw, p_away = _apply_stakes(p_home, p_draw, p_away, stakes_home, stakes_away, derby)
        derby_str = " + derby" if derby else ""
        print(f"  + Stakes : {home}={stakes_home or 'neutral'} / {away}={stakes_away or 'neutral'}{derby_str}")

    if odds_1x2:
        p_home, p_draw, p_away, implied = _apply_odds(p_home, p_draw, p_away, odds_1x2, odds_blend)
        print(f"  + Market odds (implied) : {implied['1']*100:.1f}% / {implied['X']*100:.1f}% / {implied['2']*100:.1f}% "
              f"(blend={odds_blend})")

    no_manual_input = not any([
        injured_home, injured_away, rest_days_diff,
        home_points is not None and away_points is not None,
        stakes_home, stakes_away, derby, odds_1x2,
    ])
    if no_manual_input:
        print("  (!) No manual adjustment provided (injuries, rest, standings, stakes, odds)")
        print("      -> prediction based on history only, not on the matchday context")

    print()
    print(f"  {RESULT_LABELS[2]} {home:<20} {p_home * 100:5.1f}%")
    print(f"  {RESULT_LABELS[1]:<28} {p_draw * 100:5.1f}%")
    print(f"  {RESULT_LABELS[0]} {away:<20} {p_away * 100:5.1f}%")

    if show_markets:
        _print_goal_markets(home, away, context, p_home, p_draw, p_away)

    return {"home": p_home, "draw": p_draw, "away": p_away}


if __name__ == "__main__":
    if len(sys.argv) == 3:
        predict_match(sys.argv[1], sys.argv[2])
    else:
        print('Usage: python -m src.models.predict "Home team" "Away team"')
        predict_match(
            "Man City", "Sunderland",
            injured_home=["Rodri"],
            rest_days_diff=2,
            home_position=1, home_points=68, away_position=17, away_points=31,
            odds_1x2={"1": 1.25, "X": 6.5, "2": 11.0},
        )
