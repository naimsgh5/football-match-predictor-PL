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
    "Man City", "Man United",
    injured_home=["Rodri"],
    injured_away=["Rashford"],
    rest_days_diff=0,
    home_position=1, home_points=68,
    away_position=5, away_points=60,
    stakes_home="titre",
    stakes_away="titre",
    derby=True,
    odds_1x2={"1": 1.90, "X": 3.2, "2":3.5},
    match_date="2026-01-02",
)
