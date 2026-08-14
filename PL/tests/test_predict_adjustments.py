"""Regression tests for the SIGN (not just the presence) of predict.py's post-hoc
adjustments. A real bug was found here: _apply_injuries had str_home/str_away swapped,
so a HOME team's own injury was INCREASING its win probability instead of decreasing it
(str_home/str_away are strength factors -- 1.0 = full squad -- not "severity" scores).
These tests would have caught it; keep them passing on every future change to this area."""
from src.models.predict import (
    _apply_goal_injuries,
    _apply_goal_lineup,
    _apply_injuries,
    _apply_lineup,
)

SQUAD_VALUES = {
    "Home FC": {"Home Star": 100.0, "Home Regular": 10.0},
    "Away FC": {"Away Star": 100.0, "Away Regular": 10.0},
}


def test_home_injury_lowers_home_win_probability():
    p_home, p_draw, p_away, _ = _apply_injuries(
        "Home FC", "Away FC", 0.5, 0.25, 0.25, ["Home Star"], [], SQUAD_VALUES
    )
    assert p_home < 0.5
    assert p_away > 0.25


def test_away_injury_raises_home_win_probability():
    p_home, p_draw, p_away, _ = _apply_injuries(
        "Home FC", "Away FC", 0.5, 0.25, 0.25, [], ["Away Star"], SQUAD_VALUES
    )
    assert p_home > 0.5
    assert p_away < 0.25


def test_weaker_lineup_lowers_home_win_probability():
    # fields only the cheap player, not the star -> a much weaker XI than the reference
    p_home, p_draw, p_away, _ = _apply_lineup(
        "Home FC", "Away FC", 0.5, 0.25, 0.25, ["Home Regular"], [], SQUAD_VALUES
    )
    assert p_home < 0.5


def test_home_injury_lowers_home_expected_goals():
    lh, la, _ = _apply_goal_injuries(2.0, 1.0, "Home FC", "Away FC", ["Home Star"], [], SQUAD_VALUES)
    assert lh < 2.0
    assert la > 1.0


def test_away_injury_raises_home_expected_goals():
    lh, la, _ = _apply_goal_injuries(2.0, 1.0, "Home FC", "Away FC", [], ["Away Star"], SQUAD_VALUES)
    assert lh > 2.0
    assert la < 1.0


def test_weaker_lineup_lowers_home_expected_goals():
    lh, la, _ = _apply_goal_lineup(2.0, 1.0, "Home FC", "Away FC", ["Home Regular"], [], SQUAD_VALUES)
    assert lh < 2.0
