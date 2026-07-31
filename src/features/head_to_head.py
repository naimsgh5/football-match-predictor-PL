"""Taux de victoire glissant entre les deux équipes d'un match (confrontations directes).

Note par rapport à algo/CLUBS_LOGISTIC_REGRESSION.ipynb : le calcul y mélangeait les
perspectives domicile/extérieur (le taux stocké dépendait de qui recevait lors de la
confrontation passée, pas de qui reçoit aujourd'hui). Ici l'historique est stocké du point
de vue d'une équipe de référence fixe par paire, puis reconverti côté domicile du match
courant — le taux reflète donc bien "l'équipe qui reçoit aujourd'hui a gagné X% des
confrontations passées", quel que soit qui recevait à l'époque.
"""
import numpy as np
import pandas as pd

WINDOW = 10


def add_h2h_features(df: pd.DataFrame, n: int = WINDOW):
    h2h_history: dict[tuple[str, str], list[float]] = {}
    rates = []

    for home, away, hs, aw in zip(df["home_team"], df["away_team"], df["home_score"], df["away_score"]):
        key = tuple(sorted([home, away]))
        ref_team = key[0]
        hist = h2h_history.get(key, [])
        avg_ref = np.mean(hist[-n:]) if hist else 0.5
        rates.append(avg_ref if home == ref_team else 1 - avg_ref)

        if hs > aw:
            winner = home
        elif hs == aw:
            winner = None
        else:
            winner = away

        result_ref = 0.5 if winner is None else (1.0 if winner == ref_team else 0.0)
        h2h_history.setdefault(key, []).append(result_ref)

    out = df.copy()
    out["h2h_home_win_rate"] = rates
    return out, h2h_history
