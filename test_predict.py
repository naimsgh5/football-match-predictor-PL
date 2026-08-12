"""Bac a sable pour tester des predictions a la main.

A editer librement puis lancer avec :
    E:\\conda_envs\\football-dl\\python.exe test_predict.py

(ne pas coller ce code directement dans PowerShell -- c'est du Python, pas du PowerShell ;
 il faut soit l'executer via ce fichier, soit via `python -c "..."` sur une seule ligne)
"""
from src.models.predict import predict_match

# --- Exemple 1 : modele seul, aucun ajustement manuel ------------------------------------
predict_match("Man City", "Sunderland")

# --- Exemple 2 : avec ajustements manuels -------------------------------------------------
predict_match(
    "Man City", "Sunderland",
    injured_home=["Rodri"],
    injured_away=[],
    rest_days_diff=2,
    home_position=1, home_points=68,
    away_position=17, away_points=31,
    stakes_home="titre",
    stakes_away="maintien",
    derby=False,
    odds_1x2={"1": 1.25, "X": 6.5, "2": 11.0},
    match_date="2026-01-02",
)
