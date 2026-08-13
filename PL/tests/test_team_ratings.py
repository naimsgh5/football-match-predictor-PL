"""Tests for the Dixon-Coles-style team attack/defense fit (src/features/team_ratings.py)."""
import numpy as np
import pandas as pd
import pytest

from src.features.team_ratings import expected_goals_from_ratings, fit_team_ratings

STRONG, WEAK, MID = "Strong", "Weak", "Mid"


def _synthetic_df(n_rounds: int = 20, seed: int = 0) -> pd.DataFrame:
    """A small synthetic league where STRONG genuinely outscores/outdefends WEAK, so the
    fitted ratings can be checked against a known ground truth."""
    rng = np.random.default_rng(seed)
    rows = []
    date = pd.Timestamp("2024-08-01")
    true_lambda = {
        STRONG: (2.2, 0.6),  # (scoring rate, conceding rate)
        MID: (1.3, 1.3),
        WEAK: (0.6, 2.2),
    }
    teams = [STRONG, MID, WEAK]
    for _ in range(n_rounds):
        for home, away in [(a, b) for a in teams for b in teams if a != b]:
            lam_home = (true_lambda[home][0] + true_lambda[away][1]) / 2
            lam_away = (true_lambda[away][0] + true_lambda[home][1]) / 2
            hs = rng.poisson(lam_home)
            aw = rng.poisson(lam_away)
            rows.append({"date": date, "home_team": home, "away_team": away, "home_score": hs, "away_score": aw})
            date += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def ratings():
    return fit_team_ratings(_synthetic_df())


def test_strong_team_rated_above_weak_team_at_home(ratings):
    assert ratings["attack_home"][STRONG] > ratings["attack_home"][WEAK]
    assert ratings["defense_home"][STRONG] < ratings["defense_home"][WEAK]  # lower = stingier


def test_strong_team_rated_above_weak_team_away(ratings):
    assert ratings["attack_away"][STRONG] > ratings["attack_away"][WEAK]
    assert ratings["defense_away"][STRONG] < ratings["defense_away"][WEAK]


def test_expected_goals_never_negative_or_zero(ratings):
    lh, la = expected_goals_from_ratings(STRONG, WEAK, ratings)
    assert lh > 0
    assert la > 0


def test_strong_home_favourite_outscores_weak_away_side(ratings):
    lh, la = expected_goals_from_ratings(STRONG, WEAK, ratings)
    assert lh > la


def test_unknown_team_falls_back_to_prior(ratings):
    # a team absent from the fit entirely (freshly promoted, no history) should get a
    # neutral, finite expected-goals estimate rather than crashing or defaulting to 0
    lh, la = expected_goals_from_ratings("Brand New FC", MID, ratings)
    assert lh > 0
    assert la > 0


def test_recent_matches_weighted_more_than_old_ones():
    # two seasons: STRONG dominates in the old one, WEAK dominates (recently) -- with
    # recency weighting, the fit should mostly reflect the RECENT (weak-dominant) season
    rows = []
    date = pd.Timestamp("2020-01-01")
    for _ in range(15):
        rows.append({"date": date, "home_team": STRONG, "away_team": WEAK, "home_score": 4, "away_score": 0})
        date += pd.Timedelta(days=7)
    date = pd.Timestamp("2025-01-01")
    for _ in range(15):
        rows.append({"date": date, "home_team": WEAK, "away_team": STRONG, "home_score": 4, "away_score": 0})
        date += pd.Timedelta(days=7)
    df = pd.DataFrame(rows)

    r = fit_team_ratings(df, as_of=pd.Timestamp("2025-04-01"))
    # WEAK's recent home rout should now outweigh its old away thrashings
    assert r["attack_home"][WEAK] > r["attack_home"][STRONG]
