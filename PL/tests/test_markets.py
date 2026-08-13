"""Consistency tests for the Poisson goal model (src/models/markets.py): the computed
probabilities must form valid distributions (sum to 1, etc.)."""
import numpy as np
import pytest
from scipy.stats import poisson

from src.models.markets import btts, goal_matrix, implied_1x2, over_under, top_scorelines

LAMBDA_HOME, LAMBDA_AWAY = 1.8, 1.1


@pytest.fixture
def matrix():
    return goal_matrix(LAMBDA_HOME, LAMBDA_AWAY)


def test_matrix_sums_to_one(matrix):
    # goal_matrix() renormalizes to compensate for the MAX_GOALS truncation
    assert matrix.sum() == pytest.approx(1.0)


def test_btts_yes_no_sum_to_one(matrix):
    probs = btts(matrix)
    assert probs["yes"] + probs["no"] == pytest.approx(1.0)


def test_over_under_plus_exact_line_sums_to_one(matrix):
    probs = over_under(matrix, 2.5)
    # .5 line -> no goal total lands exactly on it, over+under = everything
    assert probs["over"] + probs["under"] == pytest.approx(1.0)


def test_implied_1x2_sums_to_one(matrix):
    probs = implied_1x2(matrix)
    assert sum(probs.values()) == pytest.approx(1.0)


def test_top_scorelines_are_sorted_descending(matrix):
    scores = top_scorelines(matrix, n=5)
    probs = [p for _, _, p in scores]
    assert probs == sorted(probs, reverse=True)


def test_dixon_coles_correction_keeps_matrix_normalized():
    # rho != 0 perturbs the 4 low-score cells -- goal_matrix must still renormalize to 1
    matrix = goal_matrix(LAMBDA_HOME, LAMBDA_AWAY, rho=-0.15)
    assert matrix.sum() == pytest.approx(1.0)
    assert (matrix >= 0).all()


def test_dixon_coles_rho_zero_matches_plain_independence():
    matrix_dc = goal_matrix(LAMBDA_HOME, LAMBDA_AWAY, rho=0.0)
    home_probs = poisson.pmf(np.arange(9), LAMBDA_HOME)
    away_probs = poisson.pmf(np.arange(9), LAMBDA_AWAY)
    independent = np.outer(home_probs, away_probs)
    independent /= independent.sum()
    assert matrix_dc == pytest.approx(independent)
