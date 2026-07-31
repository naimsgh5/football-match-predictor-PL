"""Classement moyen des n dernières saisons COMPLÈTES précédant la saison du match.

Correction par rapport à algo/CLUBS_LOGISTIC_REGRESSION.ipynb : là-bas, pl_avg_rank
était calculé une fois sur TOUTES les saisons du CSV (passées et futures) puis appliqué
identiquement à tous les matchs — une fuite de données pour un usage temporel (le
classement d'un match de 2021 ne doit jamais dépendre de résultats de 2025). Ici, seules
les saisons complètes et strictement antérieures à la saison du match sont utilisées.
"""
import numpy as np
import pandas as pd

N_SEASONS = 5
FALLBACK_RANK = 20  # dernière place : pénalité pour équipe promue / sans historique


def _season_final_standings(df: pd.DataFrame) -> dict[int, dict[str, int]]:
    """Retourne {season: {team: rang_final}} à partir des matchs de chaque saison."""
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


def add_historical_rank_features(df: pd.DataFrame, n_seasons: int = N_SEASONS, fallback: int = FALLBACK_RANK):
    """Ajoute rank_home, rank_away, rank_diff : rang final moyen de l'équipe sur les
    n_seasons saisons complètes précédant strictement la saison du match courant."""
    standings = _season_final_standings(df)
    seasons_sorted = sorted(standings.keys())

    rank_home, rank_away = [], []
    for season, home, away in zip(df["season"], df["home_team"], df["away_team"]):
        past_seasons = [s for s in seasons_sorted if s < season][-n_seasons:]

        def avg_rank(team):
            ranks = [standings[s][team] for s in past_seasons if team in standings[s]]
            return np.mean(ranks) if ranks else fallback

        rank_home.append(avg_rank(home))
        rank_away.append(avg_rank(away))

    out = df.copy()
    out["rank_home"] = rank_home
    out["rank_away"] = rank_away
    out["rank_diff"] = out["rank_away"] - out["rank_home"]
    return out
