"""Marchés dérivés (scores probables, BTTS, over/under) via un modèle de buts Poisson.

Ce N'EST PAS ce qu'apprend le modèle 1X2 (elo/forme/etc. dans src/models/predict.py) -- c'est
un second modèle, indépendant et plus simple, uniquement là parce que le classifieur 1X2 ne
peut par construction pas donner de score exact ni de marché de buts (il ne prédit qu'une des
3 classes domicile/nul/extérieur). Calibré sur les moyennes glissantes de buts marqués/encaissés
déjà calculées pour attack_diff/defense_diff (src/features/rolling_stats.py::add_goals_features).

Limite connue : ces moyennes ne sont pas spécifiques au lieu (domicile/extérieur), contrairement
à venue_form_diff -- un simple multiplicateur HOME_ADVANTAGE compense approximativement l'avantage
du terrain sur le nombre de buts. Le 1X2 implicite de ce modèle de buts (implied_1x2) est affiché
à titre de recoupement avec le 1X2 du modèle principal, mais les deux peuvent diverger : ce sont
deux estimations indépendantes, pas la même prédiction recalibrée.

À titre informatif seulement -- aucune probabilité ici ne garantit un résultat.
"""
import numpy as np
from scipy.stats import poisson

MAX_GOALS = 8  # au-dela, probabilite negligeable pour un match de PL
HOME_ADVANTAGE = 1.10  # multiplicateur sur lambda_home : compense l'absence de split domicile/exterieur
OU_LINES = [1.5, 2.5, 3.5]


def expected_goals(gs_home, gc_home, gs_away, gc_away, home_advantage: float = HOME_ADVANTAGE):
    """Buts attendus (lambda Poisson) pour chaque équipe : moyenne entre l'attaque de l'une et
    la défense (fébrilité) de l'autre, comme dans un modèle Poisson classique de football."""
    lambda_home = (gs_home + gc_away) / 2 * home_advantage
    lambda_away = (gs_away + gc_home) / 2
    return max(lambda_home, 0.1), max(lambda_away, 0.1)


def goal_matrix(lambda_home: float, lambda_away: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """matrix[i, j] = P(domicile marque i, exterieur marque j), buts independants (Poisson x Poisson).
    Renormalisee pour sommer exactement a 1 (la troncature a max_goals laisse echapper une
    queue de distribution negligeable mais non nulle -- sans ca, tous les marches derives ne
    sommeraient pas a 100%)."""
    home_probs = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    away_probs = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    matrix = np.outer(home_probs, away_probs)
    return matrix / matrix.sum()


def top_scorelines(matrix: np.ndarray, n: int = 5):
    """Retourne les n scores exacts les plus probables : [(buts_domicile, buts_exterieur, proba), ...]."""
    flat_idx = np.argsort(matrix.ravel())[::-1][:n]
    rows, cols = np.unravel_index(flat_idx, matrix.shape)
    return [(int(i), int(j), float(matrix[i, j])) for i, j in zip(rows, cols)]


def btts(matrix: np.ndarray) -> dict:
    """Both Teams To Score : P(les deux equipes marquent >= 1 but)."""
    p_no = matrix[0, :].sum() + matrix[:, 0].sum() - matrix[0, 0]
    return {"yes": float(matrix.sum() - p_no), "no": float(p_no)}


def over_under(matrix: np.ndarray, line: float) -> dict:
    """P(total de buts > line) et P(total < line), pour une ligne type 2.5."""
    max_goals = matrix.shape[0] - 1
    idx = np.arange(max_goals + 1)
    i, j = np.meshgrid(idx, idx, indexing="ij")
    total = i + j
    over = float(matrix[total > line].sum())
    under = float(matrix[total < line].sum())
    return {"over": over, "under": under}


def implied_1x2(matrix: np.ndarray) -> dict:
    """1X2 implicite de ce modele de buts (a comparer, pas a confondre, avec le 1X2 du modele principal)."""
    n = matrix.shape[0]
    idx = np.arange(n)
    i, j = np.meshgrid(idx, idx, indexing="ij")
    return {
        "home": float(matrix[i > j].sum()),
        "draw": float(matrix[i == j].sum()),
        "away": float(matrix[i < j].sum()),
    }


def betting_notes(p_home: float, p_draw: float, p_away: float, btts_probs: dict, ou_probs: dict) -> list[str]:
    """Quelques observations textuelles sur les marches ou un des deux cotes ressort nettement.
    Purement informatif : ne recommande jamais un pari precis, seulement ce que dit le modele."""
    notes = []

    result_probs = {"domicile": p_home, "nul": p_draw, "exterieur": p_away}
    ranked = sorted(result_probs.items(), key=lambda kv: kv[1], reverse=True)
    best_result, best_p = ranked[0]
    gap = best_p - ranked[1][1]
    if gap >= 0.20:
        notes.append(f"Resultat : '{best_result}' domine nettement ({best_p*100:.0f}%).")
    elif gap >= 0.08:
        notes.append(f"Resultat : '{best_result}' legerement favori ({best_p*100:.0f}%), pas acquis.")
    else:
        notes.append("Resultat : aucune issue ne se degage clairement, match incertain.")

    if btts_probs["yes"] >= 0.58:
        notes.append(f"BTTS : les deux equipes marquent probablement (Oui {btts_probs['yes']*100:.0f}%).")
    elif btts_probs["no"] >= 0.58:
        notes.append(f"BTTS : au moins une equipe pourrait ne pas marquer (Non {btts_probs['no']*100:.0f}%).")
    else:
        notes.append("BTTS : pas de tendance nette.")

    line_25 = ou_probs.get(2.5)
    if line_25:
        if line_25["over"] >= 0.58:
            notes.append(f"Buts : plutot un match avec beaucoup de buts (Over 2.5 {line_25['over']*100:.0f}%).")
        elif line_25["under"] >= 0.58:
            notes.append(f"Buts : plutot un match ferme (Under 2.5 {line_25['under']*100:.0f}%).")
        else:
            notes.append("Buts : pas de tendance nette autour de 2.5.")

    notes.append(
        "A titre informatif : modele statistique simplifie, aucune garantie de resultat. "
        "Parie de maniere responsable, seulement ce que tu peux te permettre de perdre."
    )
    return notes
