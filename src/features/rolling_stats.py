"""Forme récente et buts marqués/encaissés, en moyenne glissante sur les n derniers matchs.

Calcul pré-match strict : chaque valeur pour un match donné n'utilise que les matchs
antérieurs de l'équipe concernée.
"""
import numpy as np
import pandas as pd

WINDOW = 10


def add_form_features(df: pd.DataFrame, n: int = WINDOW):
    """Ajoute form_home, form_away, form_diff (moyenne des résultats sur les n derniers matchs,
    1=victoire, 0.5=nul, 0=défaite). Retourne aussi l'historique final par équipe."""
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


def add_goals_features(df: pd.DataFrame, n: int = WINDOW, default_avg: float = 1.3):
    """Ajoute les moyennes glissantes de buts marqués/encaissés et les diffs attaque/défense.

    Le repli pour une équipe sans historique (son tout premier match dans le dataset)
    utilise la moyenne de buts cumulée des matchs déjà traités (pas la moyenne du dataset
    entier, ce qui utiliserait des matchs futurs) — avant le tout premier match, un
    default_avg fixe (~moyenne PL classique) sert d'amorce.
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
