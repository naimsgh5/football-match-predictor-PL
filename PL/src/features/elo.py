"""Pre-match Elo rating, computed in strict chronological order (no leakage from the future)."""
import pandas as pd

INITIAL_RATING = 1500
K_FACTOR = 30


def add_elo_features(df: pd.DataFrame, k: float = K_FACTOR, initial: float = INITIAL_RATING):
    """Adds elo_home, elo_away, elo_diff. df must be sorted by date.

    Returns (df_with_features, elo_final) — elo_final allows reusing the up-to-date
    ratings to predict a future match without recomputing the whole history.
    """
    elo: dict[str, float] = {}
    elo_home, elo_away = [], []

    for home, away, hs, aw in zip(df["home_team"], df["away_team"], df["home_score"], df["away_score"]):
        eh = elo.get(home, initial)
        ea = elo.get(away, initial)
        elo_home.append(eh)
        elo_away.append(ea)

        exp_home = 1 / (1 + 10 ** ((ea - eh) / 400))
        if hs > aw:
            score = 1.0
        elif hs == aw:
            score = 0.5
        else:
            score = 0.0

        elo[home] = eh + k * (score - exp_home)
        elo[away] = ea + k * ((1 - score) - (1 - exp_home))

    out = df.copy()
    out["elo_home"] = elo_home
    out["elo_away"] = elo_away
    out["elo_diff"] = out["elo_home"] - out["elo_away"]
    return out, elo
