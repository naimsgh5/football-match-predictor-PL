"""Predicts an upcoming match (never seen during training) with the two trained models
(Logistic Regression M3, MLP M4), shown separately so it's always clear which model
produced which numbers.

Recomputes the "core" features (Elo, form, home/away form, h2h, goals, 5-season rank)
from the final state of the full history, then applies optional *post-hoc* adjustments
(neither model ever learned from these, they only shift the final probability, as in
algo/CLUBS_LOGISTIC_REGRESSION.ipynb): injuries/market value, rest days, current
standings/points, bookmaker odds. The same adjustments are applied independently to
each model's raw prediction, AND to the Poisson goal market's expected goals (reshaped
as multiplicative shifts instead of additive ones -- see _apply_all_goal_adjustments)
so probable scores/BTTS/over-under react to the matchday context too, not just the 1X2s.

Usage:
    python -m src.models.predict "Arsenal" "Chelsea"
"""
import sys

import joblib
import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import RESULT_LABELS
from src.features.build_dataset import FEATURE_COLUMNS, build_dataset_with_state
from src.features.head_to_head import WINDOW as H2H_WINDOW
from src.features.historical_rank import N_SEASONS, average_rank
from src.features.market_value_injuries import injury_strength, lineup_strength
from src.features.rolling_stats import CLEAN_SHEET_DEFAULT_RATE, CONGESTION_WINDOW_DAYS, WINDOW as FORM_WINDOW
from src.features.squad_values import SQUAD_VALUES
from src.features.team_ratings import expected_goals_from_ratings
from src.models.markets import OU_LINES, btts, goal_matrix, implied_1x2, over_under, top_scorelines
from src.models.mlp import MLP
from src.models.mlp import MODEL_PATH as MLP_MODEL_PATH
from src.models.mlp import SCALER_PATH as MLP_SCALER_PATH
from src.models.mlp import predict_proba as mlp_predict_proba

LR_MODEL_PATH = "models_saved/baseline_lr.joblib"
LR_SCALER_PATH = "models_saved/baseline_lr_scaler.joblib"
DEFAULT_GOALS_AVG = 1.3
REST_DAY_IMPACT = 0.12       # maximum (asymptotic) impact of rest days on the probabilities
INJURY_IMPACT = 0.15         # maximum impact of the injury factor on the probabilities
LINEUP_IMPACT = 0.15         # maximum impact of the confirmed-lineup factor on the probabilities
STANDINGS_IMPACT = 0.15      # maximum impact of the current points gap on the probabilities
STANDINGS_POINTS_SCALE = 15  # points gap (~15 pts) beyond which the impact saturates
ODDS_BLEND_WEIGHT = 0.5      # 0 = pure model, 1 = pure odds

# Same signals, reshaped as MULTIPLICATIVE shifts on the Poisson goal model's expected goals
# (lambda_home/lambda_away) instead of additive shifts on the 1X2 probabilities above --
# without these, the goal market (probable scores/BTTS/over-under) stayed blind to every
# manual adjustment even when the 1X2 predictions clearly reacted to them (e.g. an injured
# striker would lower the home win probability but leave "expected goals" completely
# untouched). Chosen to be comparable in *order of magnitude* to the probability-impact
# constants above, not independently calibrated against real goal outcomes -- same heuristic
# status as the rest of this adjustment system.
GOALS_INJURY_IMPACT = 0.20
GOALS_LINEUP_IMPACT = 0.20
GOALS_REST_IMPACT = 0.10
GOALS_STANDINGS_IMPACT = 0.15
GOALS_STAKES_IMPACT = 2.5      # scales STAKES_LEVELS (~0.04-0.06) into a comparable goals swing
DERBY_GOALS_DAMPENING = 0.08   # a derby tends to be tighter/more cautious -- fewer goals both sides
GOALS_ODDS_IMPACT = 0.5        # nudges lambdas towards the market's home/away tilt (not an exact match --
                                # odds only encode a 1X2 signal, not a goals one, see _apply_goal_odds)


# ---------------------------------------------------------------------------
# "Core" features (same computations as at training time, applied outside the dataset)
# ---------------------------------------------------------------------------

def _team_form(team, form_history, n=FORM_WINDOW):
    hist = form_history.get(team, [])
    return np.mean(hist[-n:]) if hist else 0.5


def _quality_form(team, quality_form_history, n=FORM_WINDOW):
    hist = quality_form_history.get(team, [])
    return np.mean(hist[-n:]) if hist else 0.0


def _clean_sheet_rate(team, clean_sheet_history, n=FORM_WINDOW, default_rate=CLEAN_SHEET_DEFAULT_RATE):
    hist = clean_sheet_history.get(team, [])
    return np.mean(hist[-n:]) if hist else default_rate


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
    quality_form_diff = (
        _quality_form(home, state["quality_form_history"]) - _quality_form(away, state["quality_form_history"])
    )
    venue_form_diff = (
        _venue_form(home, state["home_venue_history"]) - _venue_form(away, state["away_venue_history"])
    )
    h2h_rate = _h2h_rate(home, away, state["h2h_history"])

    gs_home = _team_goals_avg(home, state["scored"])
    gc_home = _team_goals_avg(home, state["conceded"])
    gs_away = _team_goals_avg(away, state["scored"])
    gc_away = _team_goals_avg(away, state["conceded"])
    clean_sheet_diff = (
        _clean_sheet_rate(home, state["clean_sheet_history"]) - _clean_sheet_rate(away, state["clean_sheet_history"])
    )

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
        "quality_form_diff": quality_form_diff,
        "clean_sheet_diff": clean_sheet_diff,
    }
    context = {
        "elo_home": elo_home, "elo_away": elo_away,
        "rank_home": rank_home, "rank_away": rank_away,
        "congestion_home": congestion_home, "congestion_away": congestion_away,
        "gs_home": gs_home, "gc_home": gc_home, "gs_away": gs_away, "gc_away": gc_away,
    }
    return features, context


# ---------------------------------------------------------------------------
# Post-hoc adjustments (entered by hand, neither model knows about them)
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
    """str_home/str_away are STRENGTH factors (1.0 = full squad, lower = weakened -- see
    injury_strength), so a home injury (str_home < str_away) must LOWER p_home: delta has to
    be (str_home - str_away), not the reverse."""
    str_home, missing_home, details_home, _ = injury_strength(home, injured_home or [], squad_values)
    str_away, missing_away, details_away, _ = injury_strength(away, injured_away or [], squad_values)
    delta = (str_home - str_away) * INJURY_IMPACT
    p_home = max(0.01, p_home + delta)
    p_away = max(0.01, p_away - delta)
    total = p_home + p_draw + p_away
    info = {"home": details_home, "away": details_away}
    return p_home / total, p_draw / total, p_away / total, info


def _apply_lineup(home, away, p_home, p_draw, p_away, lineup_home, lineup_away, squad_values):
    """Complements _apply_injuries: a confirmed starting XI (lineup_home/lineup_away, 11
    names each) is compared to each team's strongest possible XI by value. Independent
    signal from injuries -- a fit player left on the bench isn't "missing", it's a
    separate rotation/tactical signal, so both adjustments can apply at once."""
    factor_home, actual_home, ref_home, unknown_home = lineup_strength(home, lineup_home or [], squad_values)
    factor_away, actual_away, ref_away, unknown_away = lineup_strength(away, lineup_away or [], squad_values)
    delta = ((factor_home - 1.0) - (factor_away - 1.0)) * LINEUP_IMPACT
    p_home = max(0.01, p_home + delta)
    p_away = max(0.01, p_away - delta)
    total = p_home + p_draw + p_away
    info = {
        "home": (factor_home, actual_home, ref_home, unknown_home),
        "away": (factor_away, actual_away, ref_away, unknown_away),
    }
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


def _apply_all_adjustments(p_home, p_draw, p_away, *, home, away, injured_home, injured_away, squad_values,
                            lineup_home, lineup_away,
                            rest_days_diff, home_points, away_points, home_position, away_position,
                            stakes_home, stakes_away, derby, odds_1x2, odds_blend):
    """Applies every requested post-hoc adjustment in sequence to a single model's raw
    (p_home, p_draw, p_away), and returns the final probabilities plus the printable
    description lines for each adjustment that was actually applied."""
    lines = []

    if injured_home or injured_away:
        p_home, p_draw, p_away, inj_info = _apply_injuries(
            home, away, p_home, p_draw, p_away, injured_home, injured_away, squad_values
        )
        lines.append(f"  + Injuries : {home} missing {list(inj_info['home'])}, {away} missing {list(inj_info['away'])}")

    if lineup_home or lineup_away:
        p_home, p_draw, p_away, lineup_info = _apply_lineup(
            home, away, p_home, p_draw, p_away, lineup_home, lineup_away, squad_values
        )
        fh, ah, rh, uh = lineup_info["home"]
        fa, aa, ra, ua = lineup_info["away"]
        home_str = f"{fh*100:.0f}% of best XI" if ah is not None else "unknown"
        away_str = f"{fa*100:.0f}% of best XI" if aa is not None else "unknown"
        lines.append(f"  + Lineup : {home} fields {home_str}, {away} fields {away_str}")

    if rest_days_diff:
        p_home, p_draw, p_away, rest_factor = _apply_rest_days(p_home, p_draw, p_away, rest_days_diff)
        lines.append(f"  + Rest : {home} {'+' if rest_days_diff >= 0 else ''}{rest_days_diff}d vs {away} "
                      f"(impact {rest_factor*100:+.1f}%)")

    if home_points is not None and away_points is not None:
        p_home, p_draw, p_away, standings_factor = _apply_current_standings(p_home, p_draw, p_away, home_points, away_points)
        pos_str = (f" ({_ordinal(home_position)} vs {_ordinal(away_position)})" if home_position and away_position else "")
        lines.append(f"  + Current standings : {home_points}pts vs {away_points}pts{pos_str} (impact {standings_factor*100:+.1f}%)")

    if stakes_home or stakes_away or derby:
        p_home, p_draw, p_away = _apply_stakes(p_home, p_draw, p_away, stakes_home, stakes_away, derby)
        derby_str = " + derby" if derby else ""
        lines.append(f"  + Stakes : {home}={stakes_home or 'neutral'} / {away}={stakes_away or 'neutral'}{derby_str}")

    if odds_1x2:
        p_home, p_draw, p_away, implied = _apply_odds(p_home, p_draw, p_away, odds_1x2, odds_blend)
        lines.append(f"  + Market odds (implied) : {implied['1']*100:.1f}% / {implied['X']*100:.1f}% / {implied['2']*100:.1f}% "
                      f"(blend={odds_blend})")

    return p_home, p_draw, p_away, lines


# ---------------------------------------------------------------------------
# Post-hoc adjustments for the GOAL market (same inputs/order as above, reshaped as
# multiplicative shifts on expected goals instead of additive shifts on 1X2 probabilities)
# ---------------------------------------------------------------------------

def _apply_goal_injuries(lambda_home, lambda_away, home, away, injured_home, injured_away, squad_values):
    """str_home/str_away are STRENGTH factors (1.0 = full squad, lower = weakened -- see
    injury_strength), so convert to a "how much weakened" signal (0 = no weakening) before
    applying it, same convention as _apply_goal_lineup below."""
    str_home, missing_home, details_home, _ = injury_strength(home, injured_home or [], squad_values)
    str_away, missing_away, details_away, _ = injury_strength(away, injured_away or [], squad_values)
    weaken_home = 1.0 - str_home
    weaken_away = 1.0 - str_away
    lambda_home *= (1 - weaken_home * GOALS_INJURY_IMPACT) * (1 + weaken_away * GOALS_INJURY_IMPACT)
    lambda_away *= (1 - weaken_away * GOALS_INJURY_IMPACT) * (1 + weaken_home * GOALS_INJURY_IMPACT)
    info = {"home": details_home, "away": details_away}
    return max(lambda_home, 0.1), max(lambda_away, 0.1), info


def _apply_goal_lineup(lambda_home, lambda_away, home, away, lineup_home, lineup_away, squad_values):
    factor_home, actual_home, ref_home, unknown_home = lineup_strength(home, lineup_home or [], squad_values)
    factor_away, actual_away, ref_away, unknown_away = lineup_strength(away, lineup_away or [], squad_values)
    weaken_home = 1.0 - factor_home
    weaken_away = 1.0 - factor_away
    lambda_home *= (1 - weaken_home * GOALS_LINEUP_IMPACT) * (1 + weaken_away * GOALS_LINEUP_IMPACT)
    lambda_away *= (1 - weaken_away * GOALS_LINEUP_IMPACT) * (1 + weaken_home * GOALS_LINEUP_IMPACT)
    info = {
        "home": (factor_home, actual_home, ref_home, unknown_home),
        "away": (factor_away, actual_away, ref_away, unknown_away),
    }
    return max(lambda_home, 0.1), max(lambda_away, 0.1), info


def _apply_goal_rest_days(lambda_home, lambda_away, rest_days_diff):
    factor = float(np.tanh(rest_days_diff / 7.0) * GOALS_REST_IMPACT)
    lambda_home *= (1 + factor)
    lambda_away *= (1 - factor)
    return max(lambda_home, 0.1), max(lambda_away, 0.1), factor


def _apply_goal_standings(lambda_home, lambda_away, home_points, away_points):
    points_diff = home_points - away_points
    factor = float(np.tanh(points_diff / STANDINGS_POINTS_SCALE) * GOALS_STANDINGS_IMPACT)
    lambda_home *= (1 + factor)
    lambda_away *= (1 - factor)
    return max(lambda_home, 0.1), max(lambda_away, 0.1), factor


def _apply_goal_stakes(lambda_home, lambda_away, stakes_home, stakes_away, derby):
    boost_home = STAKES_LEVELS.get(stakes_home, 0.0) if stakes_home else 0.0
    boost_away = STAKES_LEVELS.get(stakes_away, 0.0) if stakes_away else 0.0
    shift = (boost_home - boost_away) * GOALS_STAKES_IMPACT
    lambda_home *= (1 + shift)
    lambda_away *= (1 - shift)

    if derby:
        lambda_home *= (1 - DERBY_GOALS_DAMPENING)
        lambda_away *= (1 - DERBY_GOALS_DAMPENING)

    return max(lambda_home, 0.1), max(lambda_away, 0.1)


def _apply_goal_odds(lambda_home, lambda_away, odds_1x2):
    """Odds are a 1X2-shaped signal (no goals information in them), so this only nudges the
    lambdas in the direction the market favors -- an approximation, not an attempt to force
    the goal matrix's implied 1X2 to exactly match the market's."""
    implied = _remove_margin(odds_1x2)
    tilt = implied["1"] - implied["2"]  # -1..+1, positive = market favors home
    lambda_home *= (1 + tilt * GOALS_ODDS_IMPACT)
    lambda_away *= (1 - tilt * GOALS_ODDS_IMPACT)
    return max(lambda_home, 0.1), max(lambda_away, 0.1), implied


def _apply_all_goal_adjustments(lambda_home, lambda_away, *, home, away, injured_home, injured_away, squad_values,
                                 lineup_home, lineup_away,
                                 rest_days_diff, home_points, away_points, home_position, away_position,
                                 stakes_home, stakes_away, derby, odds_1x2, odds_blend):
    """Applies every requested post-hoc adjustment in sequence to the goal model's raw
    (lambda_home, lambda_away) -- same inputs, same order as _apply_all_adjustments, so the
    goal market reacts to the same matchday context as the 1X2 models instead of staying
    frozen at the historical-ratings estimate. odds_blend is accepted but unused (kept so
    this can be called with the exact same **adjustment_kwargs dict as the 1X2 version)."""
    lines = []

    if injured_home or injured_away:
        lambda_home, lambda_away, inj_info = _apply_goal_injuries(
            lambda_home, lambda_away, home, away, injured_home, injured_away, squad_values
        )
        lines.append(f"  + Injuries : {home} missing {list(inj_info['home'])}, {away} missing {list(inj_info['away'])}")

    if lineup_home or lineup_away:
        lambda_home, lambda_away, lineup_info = _apply_goal_lineup(
            lambda_home, lambda_away, home, away, lineup_home, lineup_away, squad_values
        )
        fh, ah, rh, uh = lineup_info["home"]
        fa, aa, ra, ua = lineup_info["away"]
        home_str = f"{fh*100:.0f}% of best XI" if ah is not None else "unknown"
        away_str = f"{fa*100:.0f}% of best XI" if aa is not None else "unknown"
        lines.append(f"  + Lineup : {home} fields {home_str}, {away} fields {away_str}")

    if rest_days_diff:
        lambda_home, lambda_away, rest_factor = _apply_goal_rest_days(lambda_home, lambda_away, rest_days_diff)
        lines.append(f"  + Rest : {home} {'+' if rest_days_diff >= 0 else ''}{rest_days_diff}d vs {away} "
                      f"(impact {rest_factor*100:+.1f}%)")

    if home_points is not None and away_points is not None:
        lambda_home, lambda_away, standings_factor = _apply_goal_standings(lambda_home, lambda_away, home_points, away_points)
        pos_str = (f" ({_ordinal(home_position)} vs {_ordinal(away_position)})" if home_position and away_position else "")
        lines.append(f"  + Current standings : {home_points}pts vs {away_points}pts{pos_str} (impact {standings_factor*100:+.1f}%)")

    if stakes_home or stakes_away or derby:
        lambda_home, lambda_away = _apply_goal_stakes(lambda_home, lambda_away, stakes_home, stakes_away, derby)
        derby_str = " + derby" if derby else ""
        lines.append(f"  + Stakes : {home}={stakes_home or 'neutral'} / {away}={stakes_away or 'neutral'}{derby_str}")

    if odds_1x2:
        lambda_home, lambda_away, implied = _apply_goal_odds(lambda_home, lambda_away, odds_1x2)
        lines.append(f"  + Market odds (implied) : {implied['1']*100:.1f}% / {implied['X']*100:.1f}% / {implied['2']*100:.1f}%")

    return lambda_home, lambda_away, lines


def _print_model_result(label, home, away, p_home, p_draw, p_away, **adjustment_kwargs):
    """Prints one model's raw prediction, the adjustments applied to it, and its final
    result -- clearly labeled so it's never confused with another model's numbers."""
    print(f"\n  === {label} ===")
    print(f"  Raw model -> Home {p_home*100:.1f}% / Draw {p_draw*100:.1f}% / Away {p_away*100:.1f}%")

    p_home, p_draw, p_away, lines = _apply_all_adjustments(p_home, p_draw, p_away, home=home, away=away, **adjustment_kwargs)
    for line in lines:
        print(line)

    print()
    print(f"  {RESULT_LABELS[2]} {home:<20} {p_home * 100:5.1f}%")
    print(f"  {RESULT_LABELS[1]:<28} {p_draw * 100:5.1f}%")
    print(f"  {RESULT_LABELS[0]} {away:<20} {p_away * 100:5.1f}%")

    return {"home": p_home, "draw": p_draw, "away": p_away}


def _print_goal_markets(home, away, state, n_scorelines=5, **adjustment_kwargs):
    """Prints probable scores / BTTS / over-under, via the Poisson goal model from
    src/models/markets.py -- INDEPENDENT of the two 1X2 models above (see markets.py
    docstring): it's a third, separate model, only here because neither classifier can
    give an exact score by construction. Raw expected goals come from each team's fitted
    attack/defense rating (src/features/team_ratings.py); the same post-hoc adjustments as
    the 1X2 models (injuries, lineup, rest, standings, stakes, odds) are then applied to
    lambda_home/lambda_away, so this market reacts to the matchday context too instead of
    staying frozen at the historical estimate."""
    lambda_home, lambda_away = expected_goals_from_ratings(home, away, state["team_ratings"])

    print(f"\n  === Derived markets: Poisson goal model (independent of the two models above) ===")
    print(f"  Raw expected goals: {home} {lambda_home:.2f} - {lambda_away:.2f} {away}")

    lambda_home, lambda_away, lines = _apply_all_goal_adjustments(
        lambda_home, lambda_away, home=home, away=away, **adjustment_kwargs
    )
    for line in lines:
        print(line)
    if lines:
        print(f"  Adjusted expected goals: {home} {lambda_home:.2f} - {lambda_away:.2f} {away}")

    matrix = goal_matrix(lambda_home, lambda_away)

    imp = implied_1x2(matrix)
    print(f"  1X2 implied by this model: Home {imp['home']*100:.1f}% / Draw {imp['draw']*100:.1f}% / "
          f"Away {imp['away']*100:.1f}% (to compare, not confuse, with the models above)")

    print(f"  Most probable scores:")
    for i, j, p in top_scorelines(matrix, n=n_scorelines):
        print(f"    {home} {i} - {j} {away}  ({p*100:.1f}%)")

    btts_probs = btts(matrix)
    print(f"  BTTS (both teams score) : Yes {btts_probs['yes']*100:.1f}% / No {btts_probs['no']*100:.1f}%")

    for line in OU_LINES:
        probs = over_under(matrix, line)
        print(f"  Over/Under {line} : Over {probs['over']*100:.1f}% / Under {probs['under']*100:.1f}%")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_match(home: str, away: str, lr_model=None, lr_scaler=None, mlp_model=None, mlp_scaler=None,
                   state=None, injured_home=None, injured_away=None,
                   lineup_home: list = None, lineup_away: list = None, squad_values=None,
                   rest_days_diff: int = None,
                   home_position: int = None, home_points: int = None,
                   away_position: int = None, away_points: int = None,
                   odds_1x2=None, odds_blend=ODDS_BLEND_WEIGHT,
                   stakes_home: str = None, stakes_away: str = None, derby: bool = False,
                   match_date=None, show_markets: bool = True):
    """Prints the prediction of each trained model separately, and returns
    {"logistic_regression": {"home": p, "draw": p, "away": p}, "mlp": {...}}.

    Computed automatically (no input needed): elo, form, opponent-strength-weighted form
    (quality_form_diff), home/away form, h2h, goals, clean sheet rate, 5-season average
    rank, fixture congestion (congestion_diff, on match_date - defaults to today; only
    counts PL matches from this dataset, not cup/European matches -- underestimates the
    true fatigue of a team competing on multiple fronts, complement with rest_days_diff
    if you know the real situation better).

    To enter yourself (neither model knows about these):
      - injured_home / injured_away: lists of absent player names
      - lineup_home / lineup_away: confirmed starting XI (11 names each), compared to each
        team's strongest possible XI by value -- complements injured_home/away rather than
        replacing it (a player being out and a player being benched are different signals);
        realistically only known ~1h before kickoff
      - rest_days_diff: home rest days - away rest days (+2 = home more rested)
      - home_position/home_points, away_position/away_points: CURRENT standings for the
        ongoing season (absent from the historical dataset until the season is over)
      - odds_1x2: optional dict {"1": home_odds, "X": draw_odds, "2": away_odds}
      - stakes_home/stakes_away: "title" / "europe" / "survival" / "neutral" (default)
      - derby: True/False -- boosts the draw probability (tight match), doesn't favor either side

    show_markets (default True): also shows probable scores / BTTS / over-under via a
    third model (Poisson on goals, independent of the two above -- see src/models/markets.py).
    All the manual adjustments above are applied to this market too (reshaped as
    multiplicative shifts on expected goals rather than additive shifts on 1X2 probabilities).
    """
    if state is None:
        _df, state = build_dataset_with_state()
    if lr_model is None:
        lr_model = joblib.load(LR_MODEL_PATH)
    if lr_scaler is None:
        lr_scaler = joblib.load(LR_SCALER_PATH)
    if mlp_model is None:
        mlp_model = MLP(input_dim=len(FEATURE_COLUMNS))
        mlp_model.load_state_dict(torch.load(MLP_MODEL_PATH))
    if mlp_scaler is None:
        mlp_scaler = joblib.load(MLP_SCALER_PATH)
    if squad_values is None:
        squad_values = SQUAD_VALUES

    for team in (home, away):
        if team not in state["elo"]:
            print(f"! '{team}' not found in the dataset (team never seen, or name misspelled).")

    features, context = compute_features(home, away, state, match_date=match_date)
    x = np.array([[features[c] for c in FEATURE_COLUMNS]])

    x_lr = lr_scaler.transform(x)
    proba_lr = dict(zip(lr_model.classes_, lr_model.predict_proba(x_lr)[0]))
    p_away_lr, p_draw_lr, p_home_lr = proba_lr.get(0, 0.0), proba_lr.get(1, 0.0), proba_lr.get(2, 0.0)

    x_mlp = mlp_scaler.transform(x).astype(np.float32)
    proba_mlp = mlp_predict_proba(mlp_model, x_mlp)[0]
    p_away_mlp, p_draw_mlp, p_home_mlp = float(proba_mlp[0]), float(proba_mlp[1]), float(proba_mlp[2])

    print(f"\n{home} (home) vs {away} (away)")
    print(f"  Elo  : {context['elo_home']:.0f} vs {context['elo_away']:.0f}")
    print(f"  Average rank over last {N_SEASONS} seasons : {context['rank_home']:.1f} vs {context['rank_away']:.1f}")
    print(f"  Fixture congestion ({CONGESTION_WINDOW_DAYS}d, PL matches only) : "
          f"{home} {context['congestion_home']} vs {away} {context['congestion_away']}")

    no_manual_input = not any([
        injured_home, injured_away, lineup_home, lineup_away, rest_days_diff,
        home_points is not None and away_points is not None,
        stakes_home, stakes_away, derby, odds_1x2,
    ])
    if no_manual_input:
        print("  (!) No manual adjustment provided (injuries, lineup, rest, standings, stakes, odds)")
        print("      -> prediction based on history only, not on the matchday context")

    adjustment_kwargs = dict(
        injured_home=injured_home, injured_away=injured_away,
        lineup_home=lineup_home, lineup_away=lineup_away, squad_values=squad_values,
        rest_days_diff=rest_days_diff, home_points=home_points, away_points=away_points,
        home_position=home_position, away_position=away_position,
        stakes_home=stakes_home, stakes_away=stakes_away, derby=derby,
        odds_1x2=odds_1x2, odds_blend=odds_blend,
    )

    result_lr = _print_model_result(
        "Model 1/2: Logistic Regression (M3)", home, away, p_home_lr, p_draw_lr, p_away_lr, **adjustment_kwargs
    )
    result_mlp = _print_model_result(
        "Model 2/2: MLP / PyTorch (M4)", home, away, p_home_mlp, p_draw_mlp, p_away_mlp, **adjustment_kwargs
    )

    if show_markets:
        _print_goal_markets(home, away, state, **adjustment_kwargs)

    return {"logistic_regression": result_lr, "mlp": result_mlp}


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
