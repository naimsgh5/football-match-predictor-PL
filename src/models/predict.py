"""Prédiction d'un match à venir (jamais vu à l'entraînement) avec le modèle déjà entraîné.

Recalcule les features "coeur" (Elo, forme, forme domicile/extérieur, h2h, buts, classement
5 ans) à partir de l'état final de l'historique complet, puis applique des ajustements
optionnels *post-hoc* (le modèle n'a jamais appris dessus, ils ne font que déplacer la
probabilité finale, comme dans algo/CLUBS_LOGISTIC_REGRESSION.ipynb) : blessures/valeur
marchande, jours de repos, classement/points actuels, cotes bookmaker.

Usage :
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

MODEL_PATH = "models_saved/baseline_lr.joblib"
SCALER_PATH = "models_saved/baseline_lr_scaler.joblib"
DEFAULT_GOALS_AVG = 1.3
REST_DAY_IMPACT = 0.12       # impact maximal (asymptotique) des jours de repos sur les probas
INJURY_IMPACT = 0.15         # impact maximal du facteur blessures sur les probas
STANDINGS_IMPACT = 0.15      # impact maximal de l'ecart de points actuel sur les probas
STANDINGS_POINTS_SCALE = 15  # ecart de points (~15 pts) au dela duquel l'impact sature
ODDS_BLEND_WEIGHT = 0.5      # 0 = modele pur, 1 = cotes pures


# ---------------------------------------------------------------------------
# Features "coeur" (mêmes calculs qu'à l'entraînement, appliqués hors dataset)
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


def _congestion(team, match_dates, target_date, window_days=CONGESTION_WINDOW_DAYS):
    """Nombre de matchs PL joues par `team` dans les `window_days` jours avant `target_date`.
    Meme limite qu'a l'entrainement : ne voit que les matchs PL de ce dataset, pas les matchs
    de coupe/Europe."""
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
    }
    return features, context


# ---------------------------------------------------------------------------
# Ajustements post-hoc (saisis à la main, le modèle ne les connaît pas)
# ---------------------------------------------------------------------------

def _apply_rest_days(p_home, p_draw, p_away, rest_days_diff):
    """rest_days_diff : jours de repos domicile - jours de repos exterieur.
    +2 = domicile a 2 jours de repos de plus, -2 = c'est l'exterieur qui est plus repose."""
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
    """home_points/away_points : points au classement actuel (saison en cours). L'ecart de
    points pousse la proba vers l'equipe la mieux classee - sature au dela de ~15 points
    d'ecart (STANDINGS_POINTS_SCALE)."""
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


# Enjeu du match : niveau categoriel -> bonus de motivation pour l'equipe concernee.
# "derby" est separe (pas un niveau par equipe) : un derby ne favorise pas un camp, il rend
# statistiquement le match plus ferme (plus de nuls / scores serres), donc gonfle p_draw.
STAKES_LEVELS = {
    "neutre": 0.00,
    "europe": 0.04,    # qualification Champions/Europa League en jeu
    "maintien": 0.05,  # lutte pour le maintien : tres motivant (peur de la relegation)
    "titre": 0.06,     # course au titre
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
# Prédiction
# ---------------------------------------------------------------------------

def predict_match(home: str, away: str, model=None, scaler=None, state=None,
                   injured_home=None, injured_away=None, squad_values=None,
                   rest_days_diff: int = None,
                   home_position: int = None, home_points: int = None,
                   away_position: int = None, away_points: int = None,
                   odds_1x2=None, odds_blend=ODDS_BLEND_WEIGHT,
                   stakes_home: str = None, stakes_away: str = None, derby: bool = False,
                   match_date=None):
    """Retourne {"home": p, "draw": p, "away": p} et affiche un résumé.

    Calculés automatiquement (aucune saisie) : elo, forme, forme domicile/exterieur, h2h,
    buts, classement moyen 5 ans, enchainement de matchs (congestion_diff, sur match_date -
    par defaut la date du jour ; ne compte que les matchs PL de ce dataset, pas les matchs
    de coupe/Europe -- sous-estime la vraie fatigue d'une equipe engagee sur plusieurs
    tableaux, complete avec rest_days_diff si tu connais mieux la situation reelle).

    À saisir toi-même (le modèle ne les connaît pas) :
      - injured_home / injured_away : listes de noms de joueurs absents
      - rest_days_diff : jours de repos domicile - exterieur (+2 = domicile plus repose)
      - home_position/home_points, away_position/away_points : classement ACTUEL saison en
        cours (absent du dataset historique tant que la saison n'est pas terminee)
      - odds_1x2 : dict optionnel {"1": cote_domicile, "X": cote_nul, "2": cote_exterieur}
      - stakes_home/stakes_away : "titre" / "europe" / "maintien" / "neutre" (defaut)
      - derby : True/False -- gonfle la proba de nul (match ferme), ne favorise aucun camp
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
            print(f"! '{team}' absent du dataset (equipe jamais rencontree ou nom mal orthographie).")

    features, context = compute_features(home, away, state, match_date=match_date)
    x = np.array([[features[c] for c in FEATURE_COLUMNS]])
    x_scaled = scaler.transform(x)
    proba = dict(zip(model.classes_, model.predict_proba(x_scaled)[0]))
    p_away, p_draw, p_home = proba.get(0, 0.0), proba.get(1, 0.0), proba.get(2, 0.0)

    print(f"\n{home} (domicile) vs {away} (exterieur)")
    print(f"  Elo  : {context['elo_home']:.0f} vs {context['elo_away']:.0f}")
    print(f"  Rang moyen {N_SEASONS} dernieres saisons : {context['rank_home']:.1f} vs {context['rank_away']:.1f}")
    print(f"  Enchainement ({CONGESTION_WINDOW_DAYS}j, matchs PL seulement) : "
          f"{home} {context['congestion_home']} vs {away} {context['congestion_away']}")
    print(f"  Modele seul -> Domicile {p_home*100:.1f}% / Nul {p_draw*100:.1f}% / Exterieur {p_away*100:.1f}%")

    if injured_home or injured_away:
        p_home, p_draw, p_away, inj_info = _apply_injuries(
            home, away, p_home, p_draw, p_away, injured_home, injured_away, squad_values
        )
        print(f"  + Blessures : {home} manque {list(inj_info['home'])}, {away} manque {list(inj_info['away'])}")

    if rest_days_diff:
        p_home, p_draw, p_away, rest_factor = _apply_rest_days(p_home, p_draw, p_away, rest_days_diff)
        print(f"  + Repos : {home} {'+' if rest_days_diff >= 0 else ''}{rest_days_diff}j vs {away} "
              f"(impact {rest_factor*100:+.1f}%)")

    if home_points is not None and away_points is not None:
        p_home, p_draw, p_away, standings_factor = _apply_current_standings(p_home, p_draw, p_away, home_points, away_points)
        pos_str = (f" ({home_position}e vs {away_position}e)" if home_position and away_position else "")
        print(f"  + Classement actuel : {home_points}pts vs {away_points}pts{pos_str} (impact {standings_factor*100:+.1f}%)")

    if stakes_home or stakes_away or derby:
        p_home, p_draw, p_away = _apply_stakes(p_home, p_draw, p_away, stakes_home, stakes_away, derby)
        derby_str = " + derby" if derby else ""
        print(f"  + Enjeu : {home}={stakes_home or 'neutre'} / {away}={stakes_away or 'neutre'}{derby_str}")

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
            rest_days_diff=2,
            home_position=1, home_points=68, away_position=17, away_points=31,
            odds_1x2={"1": 1.25, "X": 6.5, "2": 11.0},
        )
