"""Assembles the training-ready feature dataset from raw PL data.

Usage: python -m src.features.build_dataset
"""
import numpy as np
import pandas as pd

from src.features.elo import add_elo_features
from src.features.head_to_head import add_h2h_features
from src.features.historical_rank import add_historical_rank_features
from src.features.rolling_stats import (
    add_clean_sheet_features,
    add_congestion_features,
    add_form_features,
    add_goals_features,
    add_quality_form_features,
    add_venue_form_features,
)
from src.features.team_ratings import fit_team_ratings

RAW_PATH = "data/raw/premier_league_results.csv"
OUT_PATH = "data/processed/premier_league_features.parquet"

FEATURE_COLUMNS = [
    "elo_diff",
    "form_diff",
    "venue_form_diff",
    "h2h_home_win_rate",
    "attack_diff",
    "defense_diff",
    "rank_diff",
    "congestion_diff",
    "quality_form_diff",
    "clean_sheet_diff",
]


def _get_season(d: pd.Timestamp) -> int:
    return d.year if d.month >= 8 else d.year - 1


def build_dataset(raw_path: str = RAW_PATH) -> pd.DataFrame:
    df, _state = build_dataset_with_state(raw_path)
    return df


def build_dataset_with_state(raw_path: str = RAW_PATH):
    """Same as build_dataset, but also returns (df, state) where state holds the final
    history of each module (elo, form, goals, h2h, standings) — needed to compute the
    features of a future match (outside the dataset) at inference time."""
    df = pd.read_csv(raw_path, parse_dates=["date"])
    # stable sort: several matches share the same date (last matchday of a season), an
    # unstable sort would break these ties differently depending on the subset of rows
    # processed, which would make the Elo diverge (sensitive to processing order)
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    df["season"] = df["date"].apply(_get_season)

    # 2 = home win, 1 = draw, 0 = away win
    df["result"] = np.where(
        df["home_score"] > df["away_score"], 2,
        np.where(df["home_score"] == df["away_score"], 1, 0),
    )

    df, elo_final = add_elo_features(df)
    df, quality_form_history = add_quality_form_features(df)  # needs elo_home/elo_away -> after add_elo_features
    df, form_history = add_form_features(df)
    df, home_venue_history, away_venue_history = add_venue_form_features(df)
    df, scored, conceded = add_goals_features(df)
    df, clean_sheet_history = add_clean_sheet_features(df)
    df, h2h_history = add_h2h_features(df)
    df, standings = add_historical_rank_features(df)
    df, match_dates = add_congestion_features(df)

    last_match_date = {team: max(dates) for team, dates in match_dates.items()}

    # batch fit (not a per-row rolling feature): every team's attack/defense rating "as of
    # the last known match", for the Poisson goal model (src/models/markets.py) only -- see
    # src/features/team_ratings.py. Never used as a FEATURE_COLUMNS entry, so this can safely
    # use the full df without risking leakage into the 1X2 classifiers' training.
    team_ratings = fit_team_ratings(df)

    state = {
        "elo": elo_final,
        "form_history": form_history,
        "quality_form_history": quality_form_history,
        "home_venue_history": home_venue_history,
        "away_venue_history": away_venue_history,
        "scored": scored,
        "conceded": conceded,
        "clean_sheet_history": clean_sheet_history,
        "h2h_history": h2h_history,
        "standings": standings,
        "match_dates": match_dates,
        "last_match_date": last_match_date,
        "team_ratings": team_ratings,
    }
    return df, state


if __name__ == "__main__":
    df = build_dataset()
    df.to_parquet(OUT_PATH, index=False)

    print(f"{len(df)} matches, {len(FEATURE_COLUMNS)} features -> {OUT_PATH}")
    print()
    print("Result distribution (0=away, 1=draw, 2=home):")
    print(df["result"].value_counts(normalize=True).sort_index())
    print()
    print(df[["date", "home_team", "away_team", "result"] + FEATURE_COLUMNS].tail())
