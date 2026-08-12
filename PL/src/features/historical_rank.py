"""Average final rank over the n COMPLETE seasons preceding the match's season.

Fix vs algo/CLUBS_LOGISTIC_REGRESSION.ipynb: there, pl_avg_rank was computed once over
ALL seasons in the CSV (past and future) then applied identically to every match — a
data leak for a temporal use case (a 2021 match's rank must never depend on 2025
results). Here, only complete seasons strictly before the match's season are used.
"""
import numpy as np
import pandas as pd

N_SEASONS = 5
FALLBACK_RANK = 20  # last place: penalty for a promoted team / no history


def _season_final_standings(df: pd.DataFrame) -> dict[int, dict[str, int]]:
    """Returns {season: {team: final_rank}} from each season's matches."""
    standings = {}
    for season, sg in df.groupby("season"):
        teams = set(sg["home_team"]) | set(sg["away_team"])
        pts = {t: 0 for t in teams}
        gf = {t: 0 for t in teams}
        ga = {t: 0 for t in teams}
        for home, away, hs, aw in zip(sg["home_team"], sg["away_team"], sg["home_score"], sg["away_score"]):
            if hs > aw:
                pts[home] += 3
            elif hs < aw:
                pts[away] += 3
            else:
                pts[home] += 1
                pts[away] += 1
            gf[home] += hs
            ga[home] += aw
            gf[away] += aw
            ga[away] += hs
        ranked = sorted(teams, key=lambda t: (pts[t], gf[t] - ga[t], gf[t]), reverse=True)
        standings[season] = {team: pos for pos, team in enumerate(ranked, start=1)}
    return standings


def average_rank(team: str, standings: dict[int, dict[str, int]], seasons: list[int], fallback: int = FALLBACK_RANK) -> float:
    """Average rank of a team over a given list of seasons (seasons where the team is
    absent from the standings, e.g. not yet in the PL, are skipped; fallback if none found)."""
    ranks = [standings[s][team] for s in seasons if team in standings[s]]
    return float(np.mean(ranks)) if ranks else float(fallback)


def add_historical_rank_features(df: pd.DataFrame, n_seasons: int = N_SEASONS, fallback: int = FALLBACK_RANK):
    """Adds rank_home, rank_away, rank_diff: the team's average final rank over the
    n_seasons complete seasons strictly preceding the current match's season.

    Returns (df_with_features, standings) — standings allows computing the rank for a
    future match (beyond the dataset's seasons) without recomputing everything."""
    standings = _season_final_standings(df)
    seasons_sorted = sorted(standings.keys())

    rank_home, rank_away = [], []
    for season, home, away in zip(df["season"], df["home_team"], df["away_team"]):
        past_seasons = [s for s in seasons_sorted if s < season][-n_seasons:]
        rank_home.append(average_rank(home, standings, past_seasons, fallback))
        rank_away.append(average_rank(away, standings, past_seasons, fallback))

    out = df.copy()
    out["rank_home"] = rank_home
    out["rank_away"] = rank_away
    out["rank_diff"] = out["rank_away"] - out["rank_home"]
    return out, standings
