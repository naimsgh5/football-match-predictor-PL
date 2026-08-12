"""Adjustment for missing players, weighted by their share of the squad's market value.

Unlike the other modules in src/features/, this is NOT a training feature: there's no
injury history for the 1900 past matches, so there's nothing to backfill. This utility is
used at inference time, to adjust a prediction for an upcoming match whose absences are
known — same logic as algo/CLUBS_LOGISTIC_REGRESSION.ipynb (squad_values + injury_strength).

squad_values must be filled in by hand, per club: {"Arsenal": {"Bukayo Saka": 150, ...}, ...}
(values in millions of euros). See algo/CLUBS_LOGISTIC_REGRESSION.ipynb for a filled-in example.
"""

MAX_IMPACT = 0.80   # maximum share of squad value considered "missing"
IMPACT_SCALE = 0.50  # dampening applied to the model's probability


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
