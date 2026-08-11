"""Assemble le dataset de features prêt pour l'entraînement, à partir des données brutes PL.

Usage : python -m src.features.build_dataset
"""
import numpy as np
import pandas as pd

from src.features.elo import add_elo_features
from src.features.head_to_head import add_h2h_features
from src.features.historical_rank import add_historical_rank_features
from src.features.rolling_stats import add_form_features, add_goals_features

RAW_PATH = "data/raw/premier_league_results.csv"
OUT_PATH = "data/processed/premier_league_features.parquet"

FEATURE_COLUMNS = [
    "elo_diff",
    "form_diff",
    "h2h_home_win_rate",
    "attack_diff",
    "defense_diff",
    "rank_diff",
]


def _get_season(d: pd.Timestamp) -> int:
    return d.year if d.month >= 8 else d.year - 1


def build_dataset(raw_path: str = RAW_PATH) -> pd.DataFrame:
    df, _state = build_dataset_with_state(raw_path)
    return df


def build_dataset_with_state(raw_path: str = RAW_PATH):
    """Comme build_dataset, mais retourne aussi (df, state) où state contient l'historique
    final de chaque module (elo, forme, buts, h2h, classement) — nécessaire pour calculer
    les features d'un match futur (hors dataset) au moment de l'inférence."""
    df = pd.read_csv(raw_path, parse_dates=["date"])
    # tri stable : plusieurs matchs partagent la meme date (derniere journee de saison),
    # un tri instable romprait ces egalites differemment selon le sous-ensemble de lignes
    # traite, ce qui ferait diverger l'Elo (sensible a l'ordre de traitement)
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    df["season"] = df["date"].apply(_get_season)

    # 2 = victoire domicile, 1 = nul, 0 = victoire extérieur
    df["result"] = np.where(
        df["home_score"] > df["away_score"], 2,
        np.where(df["home_score"] == df["away_score"], 1, 0),
    )

    df, elo_final = add_elo_features(df)
    df, form_history = add_form_features(df)
    df, scored, conceded = add_goals_features(df)
    df, h2h_history = add_h2h_features(df)
    df, standings = add_historical_rank_features(df)

    last_match_date = {}
    for team in set(df["home_team"]) | set(df["away_team"]):
        team_dates = df.loc[(df["home_team"] == team) | (df["away_team"] == team), "date"]
        last_match_date[team] = team_dates.max()

    state = {
        "elo": elo_final,
        "form_history": form_history,
        "scored": scored,
        "conceded": conceded,
        "h2h_history": h2h_history,
        "standings": standings,
        "last_match_date": last_match_date,
    }
    return df, state


if __name__ == "__main__":
    df = build_dataset()
    df.to_parquet(OUT_PATH, index=False)

    print(f"{len(df)} matchs, {len(FEATURE_COLUMNS)} features -> {OUT_PATH}")
    print()
    print("Repartition des resultats (0=exterieur, 1=nul, 2=domicile) :")
    print(df["result"].value_counts(normalize=True).sort_index())
    print()
    print(df[["date", "home_team", "away_team", "result"] + FEATURE_COLUMNS].tail())
