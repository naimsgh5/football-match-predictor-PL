"""Bac a sable pour tester des predictions a la main.

A editer librement puis lancer avec :
    E:\\conda_envs\\football-dl\\python.exe test_predict.py

(ne pas coller ce code directement dans PowerShell -- c'est du Python, pas du PowerShell ;
 il faut soit l'executer via ce fichier, soit via `python -c "..."` sur une seule ligne)
"""
from src.models.predict import predict_match


# --- Exemple 2 : avec ajustements manuels -------------------------------------------------
predict_match(
    "Man United", "Man City",
    injured_home=["Luke Shaw"],
    injured_away=["Rodri"],
    rest_days_diff=0,
    home_position=2, home_points=65,
    away_position=1, away_points=67,
    stakes_home="titre",
    stakes_away="titre",
    derby=True,
    odds_1x2={"1": 2.2, "X": 3, "2":2.0},
    match_date="2026-01-02",
)
