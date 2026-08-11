"""Prédiction d'un match à venir (jamais vu à l'entraînement) avec le modèle déjà entraîné.

Recalcule les features (Elo, forme, h2h, buts, classement) à partir de l'état final de
l'historique complet, puis applique 3 ajustements optionnels *post-hoc* (le modèle n'a
jamais appris sur ces signaux, ils ne font que déplacer la probabilité finale, comme dans
algo/CLUBS_LOGISTIC_REGRESSION.ipynb) : blessures/valeur marchande, jours de repos, cotes
bookmaker.

Usage :
    python -m src.models.predict "Arsenal" "Chelsea"
"""
import sys
from datetime import datetime

import joblib
import numpy as np

from src.evaluation.metrics import RESULT_LABELS
from src.features.build_dataset import FEATURE_COLUMNS, build_dataset_with_state
from src.features.head_to_head import WINDOW as H2H_WINDOW
from src.features.historical_rank import N_SEASONS, average_rank
from src.features.market_value_injuries import injury_strength
from src.features.rolling_stats import WINDOW as FORM_WINDOW
from src.features.squad_values import SQUAD_VALUES

MODEL_PATH = "models_saved/baseline_lr.joblib"
SCALER_PATH = "models_saved/baseline_lr_scaler.joblib"
DEFAULT_GOALS_AVG = 1.3
REST_DAY_IMPACT = 0.12   # impact maximal (asymptotique) des jours de repos sur les probas
INJURY_IMPACT = 0.15     # impact maximal du facteur blessures sur les probas
ODDS_BLEND_WEIGHT = 0.5  # 0 = modele pur, 1 = cotes pures


# ---------------------------------------------------------------------------
# Features "coeur" (mêmes calculs qu'à l'entraînement, appliqués hors dataset)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Ajustements post-hoc
# ---------------------------------------------------------------------------

def _rest_days_diff(home, away, state, match_date):
    """rest_home - rest_away, en jours. None si une des deux equipes n'a pas d'historique."""
    last_home = state["last_match_date"].get(home)
    last_away = state["last_match_date"].get(away)
    if last_home is None or last_away is None:
        return None
    rest_home = (match_date - last_home).days
    rest_away = (match_date - last_away).days
    return rest_home - rest_away


def _apply_rest_days(p_home, p_draw, p_away, rest_days_diff):
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


# ---------------------------------------------------------------------------
# Prédiction
# ---------------------------------------------------------------------------

def predict_match(home: str, away: str, model=None, scaler=None, state=None,
                   injured_home=None, injured_away=None, squad_values=None,
                   match_date=None, odds_1x2=None, odds_blend=ODDS_BLEND_WEIGHT):
    """Retourne {"home": p, "draw": p, "away": p} et affiche un résumé.

    injured_home/injured_away : listes de noms de joueurs absents.
    squad_values : dict {equipe: {joueur: valeur_M€}} — SQUAD_VALUES (src/features/squad_values.py) par defaut.
    match_date : date du match (defaut : aujourd'hui) — sert a calculer les jours de repos.
    odds_1x2 : dict optionnel {"1": cote_domicile, "X": cote_nul, "2": cote_exterieur}.
    """
    if state is None:
        _df, state = build_dataset_with_state()
    if model is None:
        model = joblib.load(MODEL_PATH)
    if scaler is None:
        scaler = joblib.load(SCALER_PATH)
    if squad_values is None:
        squad_values = SQUAD_VALUES
    if match_date is None:
        match_date = datetime.now()

    for team in (home, away):
        if team not in state["elo"]:
            print(f"! '{team}' absent du dataset (equipe jamais rencontree ou nom mal orthographie).")

    features, context = compute_features(home, away, state)
    x = np.array([[features[c] for c in FEATURE_COLUMNS]])
    x_scaled = scaler.transform(x)
    proba = dict(zip(model.classes_, model.predict_proba(x_scaled)[0]))
    p_away, p_draw, p_home = proba.get(0, 0.0), proba.get(1, 0.0), proba.get(2, 0.0)

    print(f"\n{home} (domicile) vs {away} (exterieur)")
    print(f"  Elo  : {context['elo_home']:.0f} vs {context['elo_away']:.0f}")
    print(f"  Rang moyen {N_SEASONS} dernieres saisons : {context['rank_home']:.1f} vs {context['rank_away']:.1f}")
    print(f"  Modele seul -> Domicile {p_home*100:.1f}% / Nul {p_draw*100:.1f}% / Exterieur {p_away*100:.1f}%")

    if injured_home or injured_away:
        p_home, p_draw, p_away, inj_info = _apply_injuries(
            home, away, p_home, p_draw, p_away, injured_home, injured_away, squad_values
        )
        print(f"  + Blessures : {home} manque {list(inj_info['home'])}, {away} manque {list(inj_info['away'])}")

    rest_diff = _rest_days_diff(home, away, state, match_date)
    if rest_diff is not None and rest_diff != 0:
        p_home, p_draw, p_away, rest_factor = _apply_rest_days(p_home, p_draw, p_away, rest_diff)
        print(f"  + Repos : {home} {'+' if rest_diff >= 0 else ''}{rest_diff}j vs {away} (impact {rest_factor*100:+.1f}%)")

    if odds_1x2:
        p_home, p_draw, p_away, implied = _apply_odds(p_home, p_draw, p_away, odds_1x2, odds_blend)
        print(f"  + Cotes marche (implicite) : {implied['1']*100:.1f}% / {implied['X']*100:.1f}% / {implied['2']*100:.1f}% "
              f"(blend={odds_blend})")

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
        predict_match(
            "Man City", "Sunderland",
            injured_home=["Rodri"],
            odds_1x2={"1": 1.25, "X": 6.5, "2": 11.0},
        )
