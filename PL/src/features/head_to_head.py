"""Rolling win rate between the two teams in a match (head-to-head).

Note vs algo/CLUBS_LOGISTIC_REGRESSION.ipynb: that version mixed up the home/away
perspective (the stored rate depended on who was hosting in the past meeting, not who
is hosting today). Here the history is stored from the point of view of a fixed
reference team per pair, then converted back to the current match's home side — so the
rate correctly reflects "the team hosting today won X% of past meetings", regardless
of who was hosting back then.
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
