"""Consistency tests for the Poisson goal model (src/models/markets.py): the computed
probabilities must form valid distributions (sum to 1, etc.)."""
import numpy as np
import pytest

from src.models.markets import btts, expected_goals, goal_matrix, implied_1x2, over_under, top_scorelines

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


def test_expected_goals_never_negative_or_zero():
    lh, la = expected_goals(gs_home=0.0, gc_home=0.0, gs_away=0.0, gc_away=0.0)
    assert lh > 0
    assert la > 0


def test_home_advantage_increases_home_expected_goals():
    lh_with_adv, _ = expected_goals(1.5, 1.0, 1.2, 1.0, home_advantage=1.10)
    lh_no_adv, _ = expected_goals(1.5, 1.0, 1.2, 1.0, home_advantage=1.0)
    assert lh_with_adv > lh_no_adv
