"""Ajustement pour joueurs absents, pondéré par leur poids dans la valeur marchande de l'effectif.

Contrairement aux autres modules de src/features/, ceci n'est PAS une feature d'entraînement :
on n'a pas d'historique des blessures pour les 1900 matchs passés, donc rien à backfill. Cet
utilitaire sert à l'inférence, pour ajuster une prédiction sur un match à venir dont on connaît
les absences — même logique que algo/CLUBS_LOGISTIC_REGRESSION.ipynb (squad_values + injury_strength).

squad_values doit être rempli à la main, par équipe : {"Arsenal": {"Bukayo Saka": 150, ...}, ...}
(valeurs en millions d'euros). Voir algo/CLUBS_LOGISTIC_REGRESSION.ipynb pour un exemple rempli.
"""

MAX_IMPACT = 0.80   # part maximale de la valeur d'effectif considérée comme "absente"
IMPACT_SCALE = 0.50  # atténuation appliquée à la probabilité du modèle


def injury_strength(team: str, injured_players: list[str], squad_values: dict[str, dict[str, float]]):
    """Retourne (facteur_force in [0.5, 1], valeur_absente_M€, détail par joueur, joueurs inconnus).

    facteur_force multiplie le xG/la probabilité de victoire de l'équipe : 1.0 = effectif complet,
    0.5 = pire cas (80% de la valeur d'effectif absente, plafonné).
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
