"""Derived markets (probable scores, BTTS, over/under) via a Poisson goal model.

This is NOT what the 1X2 model (elo/form/etc. in src/models/predict.py) learns -- it's a
second, independent, simpler model, only here because the 1X2 classifier can't by
construction give an exact score or a goals market (it only predicts one of 3 classes:
home/draw/away). Calibrated on the rolling averages of goals scored/conceded already
computed for attack_diff/defense_diff (src/features/rolling_stats.py::add_goals_features).

Known limitation: these averages aren't venue-specific (home/away), unlike venue_form_diff
-- a simple HOME_ADVANTAGE multiplier approximately compensates for home-field advantage on
goal counts. This goal model's implied 1X2 (implied_1x2) is shown as a cross-check against
the main model's 1X2, but the two can diverge: they're two independent estimates, not the
same prediction recalibrated.

For informational purposes only -- no probability here guarantees a result.
"""
import numpy as np
from scipy.stats import poisson

MAX_GOALS = 8  # beyond this, negligible probability for a PL match
HOME_ADVANTAGE = 1.10  # multiplier on lambda_home: compensates for the lack of a home/away split
OU_LINES = [1.5, 2.5, 3.5]


def expected_goals(gs_home, gc_home, gs_away, gc_away, home_advantage: float = HOME_ADVANTAGE):
    """Expected goals (Poisson lambda) for each team: average of one team's attack and the
    other's defense (leakiness), as in a classic football Poisson model."""
    lambda_home = (gs_home + gc_away) / 2 * home_advantage
    lambda_away = (gs_away + gc_home) / 2
    return max(lambda_home, 0.1), max(lambda_away, 0.1)


def goal_matrix(lambda_home: float, lambda_away: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """matrix[i, j] = P(home scores i, away scores j), independent goals (Poisson x Poisson).
    Renormalized to sum to exactly 1 (truncating at max_goals lets a negligible but non-zero
    tail of the distribution escape -- without this, the derived markets wouldn't sum to 100%)."""
    home_probs = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    away_probs = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    matrix = np.outer(home_probs, away_probs)
    return matrix / matrix.sum()


def top_scorelines(matrix: np.ndarray, n: int = 5):
    """Returns the n most probable exact scores: [(home_goals, away_goals, proba), ...]."""
    flat_idx = np.argsort(matrix.ravel())[::-1][:n]
    rows, cols = np.unravel_index(flat_idx, matrix.shape)
    return [(int(i), int(j), float(matrix[i, j])) for i, j in zip(rows, cols)]


def btts(matrix: np.ndarray) -> dict:
    """Both Teams To Score: P(both teams score >= 1 goal)."""
    p_no = matrix[0, :].sum() + matrix[:, 0].sum() - matrix[0, 0]
    return {"yes": float(matrix.sum() - p_no), "no": float(p_no)}


def over_under(matrix: np.ndarray, line: float) -> dict:
    """P(total goals > line) and P(total < line), for a line like 2.5."""
    max_goals = matrix.shape[0] - 1
    idx = np.arange(max_goals + 1)
    i, j = np.meshgrid(idx, idx, indexing="ij")
    total = i + j
    over = float(matrix[total > line].sum())
    under = float(matrix[total < line].sum())
    return {"over": over, "under": under}


def implied_1x2(matrix: np.ndarray) -> dict:
    """This goal model's implied 1X2 (to compare, not confuse, with the main model's 1X2)."""
    n = matrix.shape[0]
    idx = np.arange(n)
    i, j = np.meshgrid(idx, idx, indexing="ij")
    return {
        "home": float(matrix[i > j].sum()),
        "draw": float(matrix[i == j].sum()),
        "away": float(matrix[i < j].sum()),
    }

