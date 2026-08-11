"""Prédiction d'un match à venir (jamais vu à l'entraînement) avec le modèle déjà entraîné.

Recalcule les features (Elo, forme, h2h, buts, classement) à partir de l'état final de
l'historique complet, exactement comme à l'entraînement mais pour un match hors dataset.

Usage :
    python -m src.models.predict "Arsenal" "Chelsea"
"""
import sys

import joblib
import numpy as np

from src.evaluation.metrics import RESULT_LABELS
from src.features.build_dataset import FEATURE_COLUMNS, build_dataset_with_state
from src.features.head_to_head import WINDOW as H2H_WINDOW
from src.features.historical_rank import N_SEASONS, average_rank
from src.features.market_value_injuries import injury_strength
from src.features.rolling_stats import WINDOW as FORM_WINDOW

MODEL_PATH = "models_saved/baseline_lr.joblib"
SCALER_PATH = "models_saved/baseline_lr_scaler.joblib"
DEFAULT_GOALS_AVG = 1.3


def _team_form(team, form_history, n=FORM_WINDOW):
    hist = form_history.get(team, [])
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


def compute_features(home: str, away: str, state: dict):
    """Reproduit exactement la logique de src/features/ pour un match hors dataset."""
    elo_home = state["elo"].get(home, 1500)
    elo_away = state["elo"].get(away, 1500)

    form_diff = _team_form(home, state["form_history"]) - _team_form(away, state["form_history"])
    h2h_rate = _h2h_rate(home, away, state["h2h_history"])

    gs_home = _team_goals_avg(home, state["scored"])
    gc_home = _team_goals_avg(home, state["conceded"])
    gs_away = _team_goals_avg(away, state["scored"])
    gc_away = _team_goals_avg(away, state["conceded"])

    rank_diff, rank_home, rank_away = _rank_diff(home, away, state["standings"])

    features = {
        "elo_diff": elo_home - elo_away,
        "form_diff": form_diff,
        "h2h_home_win_rate": h2h_rate,
        "attack_diff": gs_home - gs_away,
        "defense_diff": gc_away - gc_home,
        "rank_diff": rank_diff,
    }
    context = {"elo_home": elo_home, "elo_away": elo_away, "rank_home": rank_home, "rank_away": rank_away}
    return features, context


def predict_match(home: str, away: str, model=None, scaler=None, state=None,
                   injured_home=None, injured_away=None, squad_values=None):
    """Retourne {"home": p, "draw": p, "away": p} et affiche un résumé.

    injured_home/injured_away + squad_values : ajustement post-hoc optionnel (comme
    algo/CLUBS_LOGISTIC_REGRESSION.ipynb) -- le modèle n'a jamais appris sur des blessures,
    donc ceci ne fait que déplacer les probabilités, pas les recalculer proprement.
    """
    if state is None:
        _df, state = build_dataset_with_state()
    if model is None:
        model = joblib.load(MODEL_PATH)
    if scaler is None:
        scaler = joblib.load(SCALER_PATH)

    for team in (home, away):
        if team not in state["elo"]:
            print(f"! '{team}' absent du dataset (equipe jamais rencontree ou nom mal orthographie).")

    features, context = compute_features(home, away, state)
    x = np.array([[features[c] for c in FEATURE_COLUMNS]])
    x_scaled = scaler.transform(x)
    proba = dict(zip(model.classes_, model.predict_proba(x_scaled)[0]))
    p_away, p_draw, p_home = proba.get(0, 0.0), proba.get(1, 0.0), proba.get(2, 0.0)

    if squad_values and (injured_home or injured_away):
        str_home, _, _, _ = injury_strength(home, injured_home or [], squad_values)
        str_away, _, _, _ = injury_strength(away, injured_away or [], squad_values)
        delta = (str_away - str_home) * 0.15
        p_home = max(0.01, p_home + delta)
        p_away = max(0.01, p_away - delta)
        total = p_home + p_draw + p_away
        p_home, p_draw, p_away = p_home / total, p_draw / total, p_away / total

    print(f"\n{home} (domicile) vs {away} (exterieur)")
    print(f"  Elo  : {context['elo_home']:.0f} vs {context['elo_away']:.0f}")
    print(f"  Rang moyen {N_SEASONS} dernieres saisons : {context['rank_home']:.1f} vs {context['rank_away']:.1f}")
    print()
    print(f"  {RESULT_LABELS[2]} {home:<20} {p_home * 100:5.1f}%")
    print(f"  {RESULT_LABELS[1]:<28} {p_draw * 100:5.1f}%")
    print(f"  {RESULT_LABELS[0]} {away:<20} {p_away * 100:5.1f}%")

    return {"home": p_home, "draw": p_draw, "away": p_away}


if __name__ == "__main__":
    if len(sys.argv) == 3:
        predict_match(sys.argv[1], sys.argv[2])
    else:
        print('Usage: python -m src.models.predict "Equipe domicile" "Equipe exterieur"')
        predict_match("Arsenal", "Chelsea")
