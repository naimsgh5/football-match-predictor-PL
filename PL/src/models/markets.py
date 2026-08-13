"""Derived markets (probable scores, BTTS, over/under) via a Poisson goal model.

This is NOT what the 1X2 model (elo/form/etc. in src/models/predict.py) learns -- it's a
second, independent, simpler model, only here because the 1X2 classifier can't by
construction give an exact score or a goals market (it only predicts one of 3 classes:
home/draw/away). Expected goals (lambda_home/lambda_away) come from
src/features/team_ratings.py -- a Dixon-Coles/Maher-style joint fit of every team's
attack/defense strength, split by venue and adjusted for the strength of opponents actually
faced (unlike a simple rolling average of goals scored/conceded, which can't tell a team
that's genuinely strong from one that just played weak opposition).

DIXON_COLES_RHO applies the low-score correlation correction from Dixon & Coles (1997) to the
4 low-scoring cells of the goal matrix (0-0, 1-0, 0-1, 1-1): plain independent Poisson (P(home
scores i) x P(away scores j)) is known to be biased there specifically, because real matches
with few goals aren't perfectly independent event-by-event (a team already up 1-0 tends to
sit back, reducing the away team's remaining chances, etc). This is a separate, complementary
correction to the attack/defense ratings above -- it corrects the independence ASSUMPTION
between the two teams' goal counts, not the expected values themselves. Calibrated per league
(max log-likelihood on that league's historical low-scoring matches, using each match's
team_ratings-implied lambda) -- see calibrate_rho.py.

This goal model's implied 1X2 (implied_1x2) is shown as a cross-check against the main
model's 1X2, but the two can diverge: they're two independent estimates, not the same
prediction recalibrated.

For informational purposes only -- no probability here guarantees a result.
"""
import numpy as np
from scipy.stats import poisson

MAX_GOALS = 8  # beyond this, negligible probability for a PL match
DIXON_COLES_RHO = -0.030  # calibrated on data/raw/premier_league_results.csv (see module docstring)
OU_LINES = [1.5, 2.5, 3.5]


def _dixon_coles_tau(x: int, y: int, lambda_home: float, lambda_away: float, rho: float) -> float:
    """Dixon & Coles (1997) correction factor for the 4 low-scoring cells; 1.0 (no change)
    everywhere else. Multiplies the independent Poisson probability at (x, y)."""
    if x == 0 and y == 0:
        return 1 - lambda_home * lambda_away * rho
    if x == 0 and y == 1:
        return 1 + lambda_home * rho
    if x == 1 and y == 0:
        return 1 + lambda_away * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def goal_matrix(lambda_home: float, lambda_away: float, max_goals: int = MAX_GOALS,
                 rho: float = DIXON_COLES_RHO) -> np.ndarray:
    """matrix[i, j] = P(home scores i, away scores j): independent Poisson (Poisson x
    Poisson), corrected on the 4 low-scoring cells by the Dixon-Coles factor (rho=0 disables
    it, recovering plain independence). Renormalized to sum to exactly 1 (truncating at
    max_goals lets a negligible but non-zero tail of the distribution escape, and the
    Dixon-Coles correction itself slightly shifts the total mass -- without renormalizing,
    the derived markets wouldn't sum to 100%)."""
    home_probs = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    away_probs = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    matrix = np.outer(home_probs, away_probs)

    for i, j in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        matrix[i, j] *= _dixon_coles_tau(i, j, lambda_home, lambda_away, rho)
    matrix = np.clip(matrix, 0, None)  # guard against a pathological rho pushing a cell negative

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
