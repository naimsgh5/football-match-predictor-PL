"""Sandbox for testing predictions by hand.

Edit freely then run with:
    E:\\conda_envs\\football-dl\\python.exe test_predict.py

(don't paste this code directly into PowerShell -- it's Python, not PowerShell;
 either run it via this file, or via `python -c "..."` on a single line)
"""
from src.models.predict import predict_match


# --- Example: with manual adjustments -------------------------------------------------
predict_match(
    "Man United", "Man City",
    injured_home=["Luke Shaw"],
    injured_away=["Rodri"],
    rest_days_diff=0,
    home_position=2, home_points=65,
    away_position=1, away_points=67,
    stakes_home="title",
    stakes_away="title",
    derby=True,
    odds_1x2={"1": 2.2, "X": 3, "2": 2.0},
    match_date="2026-01-02",
)
