"""Team-level attack/defense strength for the Poisson goal model (src/models/markets.py) --
NOT a training feature for the 1X2 classifiers (LR/MLP), which keep learning from
src/features/rolling_stats.py's rolling averages instead.

Unlike a simple rolling average of goals scored/conceded, this adjusts for the strength of
the opponents actually faced: two teams can have the same average goals scored while one
played much tougher opposition -- a raw average can't tell them apart (the same problem
quality_form_diff solves for the 1X2 models' form_diff, addressed here for the goal model).

Fits 4 log-scale parameters per team -- attack_home, defense_home, attack_away, defense_away
-- by maximizing a Poisson log-likelihood jointly over every match at once (a Dixon-Coles /
Maher-style model): expected goals for team A hosting team B is
exp(attack_home[A] + defense_away[B]), and symmetrically exp(attack_away[B] + defense_home[A])
for B. Splitting attack/defense by venue (rather than the single rating + shared home-advantage
constant of the original 1997 Dixon-Coles paper) lets the model directly represent a team
that's simply hard to beat at home without necessarily being high-scoring there -- the
motivating case for this over both a flat home-advantage multiplier and a raw venue-specific
average.

Two safeguards against overfitting on ~19-23 home matches/team/season:
  - RIDGE_LAMBDA: L2 shrinkage of all 4 ratings towards a shared prior (the value that makes
    an unfitted team default to DEFAULT_GOALS_AVG expected goals) -- also what makes the
    optimization problem identifiable in the first place (attack/defense enter the model only
    as a sum, so without shrinkage any constant could shift from one to the other freely).
  - HALF_LIFE_DAYS: each match is weighted by its recency (exponential decay) when fitting --
    a team's true strength isn't stable over 5 seasons of transfers and manager changes, so
    a 2021 result shouldn't count as much as one from last month.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

HALF_LIFE_DAYS = 365.0  # a match this old counts half as much as one from today
RIDGE_LAMBDA = 0.01     # L2 shrinkage strength towards the prior (see module docstring)
DEFAULT_GOALS_AVG = 1.3  # ~typical top-flight average; what an unrated team defaults to


def fit_team_ratings(df: pd.DataFrame, as_of: pd.Timestamp = None,
                      half_life_days: float = HALF_LIFE_DAYS, ridge_lambda: float = RIDGE_LAMBDA,
                      default_goals_avg: float = DEFAULT_GOALS_AVG) -> dict:
    """Fits attack_home/defense_home/attack_away/defense_away for every team that appears in
    df, using all matches up to `as_of` (defaults to the last match date in df -- i.e. "as of
    right now", for predicting a future match; never call this to score matches that are
    themselves inside df, that would leak each match's own result into its own rating).

    Returns {"attack_home": {team: float}, "defense_home": {...}, "attack_away": {...},
    "defense_away": {...}, "prior": float} -- prior is the fallback rating for a team absent
    from df entirely (freshly promoted, no history yet)."""
    if as_of is None:
        as_of = df["date"].max()

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    days_ago = (as_of - df["date"]).dt.days.clip(lower=0).to_numpy(dtype=float)
    weights = 0.5 ** (days_ago / half_life_days)

    h = df["home_team"].map(idx).to_numpy()
    a = df["away_team"].map(idx).to_numpy()
    x = df["home_score"].to_numpy(dtype=float)
    y = df["away_score"].to_numpy(dtype=float)

    prior = np.log(default_goals_avg) / 2  # attack + defense both at prior -> lambda = default_goals_avg

    def unpack(theta):
        return theta[0:n], theta[n:2 * n], theta[2 * n:3 * n], theta[3 * n:4 * n]

    def objective_and_grad(theta):
        attack_home, defense_home, attack_away, defense_away = unpack(theta)
        log_lam_home = attack_home[h] + defense_away[a]
        log_lam_away = attack_away[a] + defense_home[h]
        lam_home = np.exp(log_lam_home)
        lam_away = np.exp(log_lam_away)

        ll = weights * (x * log_lam_home - lam_home + y * log_lam_away - lam_away)
        reg = ridge_lambda * np.sum((theta - prior) ** 2)
        nll = -ll.sum() + reg

        resid_home = weights * (x - lam_home)  # d(log-lik)/d(log_lam_home)
        resid_away = weights * (y - lam_away)  # d(log-lik)/d(log_lam_away)

        grad = np.empty_like(theta)
        grad[0:n] = -np.bincount(h, weights=resid_home, minlength=n)          # attack_home
        grad[n:2 * n] = -np.bincount(h, weights=resid_away, minlength=n)      # defense_home
        grad[2 * n:3 * n] = -np.bincount(a, weights=resid_away, minlength=n)  # attack_away
        grad[3 * n:4 * n] = -np.bincount(a, weights=resid_home, minlength=n)  # defense_away
        grad += 2 * ridge_lambda * (theta - prior)

        return nll, grad

    theta0 = np.full(4 * n, prior)
    result = minimize(objective_and_grad, theta0, jac=True, method="L-BFGS-B")
    attack_home, defense_home, attack_away, defense_away = unpack(result.x)

    return {
        "attack_home": {t: float(attack_home[i]) for t, i in idx.items()},
        "defense_home": {t: float(defense_home[i]) for t, i in idx.items()},
        "attack_away": {t: float(attack_away[i]) for t, i in idx.items()},
        "defense_away": {t: float(defense_away[i]) for t, i in idx.items()},
        "prior": float(prior),
    }


def expected_goals_from_ratings(home: str, away: str, ratings: dict):
    """Expected goals (Poisson lambda) for each team, from fitted ratings -- see
    fit_team_ratings. Teams absent from the fit (no history at all) fall back to the shared
    prior (~DEFAULT_GOALS_AVG expected goals, a neutral "average team" assumption)."""
    prior = ratings.get("prior", 0.0)
    attack_home = ratings["attack_home"].get(home, prior)
    defense_away = ratings["defense_away"].get(away, prior)
    attack_away = ratings["attack_away"].get(away, prior)
    defense_home = ratings["defense_home"].get(home, prior)

    lambda_home = np.exp(attack_home + defense_away)
    lambda_away = np.exp(attack_away + defense_home)
    return max(float(lambda_home), 0.1), max(float(lambda_away), 0.1)
