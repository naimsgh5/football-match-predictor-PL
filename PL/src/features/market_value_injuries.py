"""Adjustment for missing players, weighted by their share of the squad's market value.

Unlike the other modules in src/features/, this is NOT a training feature: there's no
injury history for the 1900 past matches, so there's nothing to backfill. This utility is
used at inference time, to adjust a prediction for an upcoming match whose absences are
known — same logic as algo/CLUBS_LOGISTIC_REGRESSION.ipynb (squad_values + injury_strength).

squad_values must be filled in by hand, per club: {"Arsenal": {"Bukayo Saka": 150, ...}, ...}
(values in millions of euros). See algo/CLUBS_LOGISTIC_REGRESSION.ipynb for a filled-in example.
"""

MAX_IMPACT = 0.80    # maximum share of squad value considered "missing"
IMPACT_SCALE = 0.50   # dampening applied to the model's probability
REFERENCE_XI_SIZE = 11


def injury_strength(team: str, injured_players: list[str], squad_values: dict[str, dict[str, float]]):
    """Returns (strength_factor in [0.5, 1], missing_value_M€, per-player detail, unknown players).

    strength_factor multiplies the team's win xG/probability: 1.0 = full squad,
    0.5 = worst case (80% of squad value missing, capped).
    """
    squad = squad_values.get(team, {})
    if not squad or not injured_players:
        return 1.0, 0.0, {}, []

    total = sum(squad.values())
    details = {p: squad[p] for p in injured_players if p in squad}
    unknown = [p for p in injured_players if p not in squad]
    missing = sum(details.values())
    frac = min(missing / total, MAX_IMPACT)
    return 1.0 - frac * IMPACT_SCALE, missing, details, unknown


def reference_xi_value(team: str, squad_values: dict[str, dict[str, float]], n: int = REFERENCE_XI_SIZE) -> float:
    """Sum of the team's n most valuable players -- the strongest lineup theoretically
    possible, used as a baseline against which a confirmed starting XI is compared."""
    squad = squad_values.get(team, {})
    top_n = sorted(squad.values(), reverse=True)[:n]
    return sum(top_n)


def lineup_strength(team: str, starting_xi: list[str], squad_values: dict[str, dict[str, float]]):
    """Compares a confirmed starting XI to the team's strongest possible XI (by market
    value). Complements injury_strength() rather than replacing it: an absent (injured/
    suspended) player reduces the squad's overall available quality, while a fit player
    left on the bench is a separate signal (tactical rotation, squad depth) -- a player
    being out and a player being benched are not the same thing, so both are tracked.

    Returns (strength_factor, actual_value_M€, reference_value_M€, unknown_players).
    strength_factor centered at 1.0 (full-strength XI fielded), lower if the fielded XI is
    worth less than the strongest possible one. No position-awareness (squad_values.py
    doesn't track positions) -- "strongest 11 by value" approximates a real best XI, it
    isn't one.
    """
    squad = squad_values.get(team, {})
    if not squad or not starting_xi:
        return 1.0, None, None, []

    reference = reference_xi_value(team, squad_values)
    known = [squad[p] for p in starting_xi if p in squad]
    unknown = [p for p in starting_xi if p not in squad]
    actual = sum(known)

    factor = (actual / reference) if reference else 1.0
    return factor, actual, reference, unknown
