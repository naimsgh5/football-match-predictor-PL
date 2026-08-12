# Useful commands

Cheat sheet of common commands for the Premier League project. **Everything runs from
this folder** (`e:\PROJET DL\PL`):

```powershell
cd "e:\PROJET DL\PL"
```

Using the Python interpreter from the conda environment on `E:` (drive `C:` is nearly
full, see README):

```
E:\conda_envs\football-dl\python.exe
```

To avoid retyping it every time, you can set it temporarily for your PowerShell session:
```powershell
$py = "E:\conda_envs\football-dl\python.exe"
```
then use `& $py ...` in the commands below (otherwise, replace `python` with the full
path each time).

---

## Update the data

The dataset (`data/raw/premier_league_results.csv`) is updated **by hand** (no
automation yet):

1. Re-download the latest Premier League results (e.g. from football-data.co.uk, the
   same source as `data/raw/E0_2025_26_footballdata.csv`)
2. Merge the new matches into `data/raw/premier_league_results.csv` (same method as
   in M1: no duplicates, no score discrepancies on matches already present)
3. Regenerate the feature dataset:

```powershell
E:\conda_envs\football-dl\python.exe -m src.features.build_dataset
```

Recommended frequency: once a week during the season (teams play ~1x/week), and
especially **right before** predicting a specific match, so the history is up to date.

---

## Check everything still works (tests)

```powershell
E:\conda_envs\football-dl\python.exe -m pytest tests/ -v
```

Rerun after any change to the features or the dataset (anti-lookahead-leakage,
Poisson market consistency, etc.).

---

## Retrain the models

```powershell
# Baseline Logistic Regression (M3)
E:\conda_envs\football-dl\python.exe -m src.models.baseline_lr

# MLP PyTorch (M4)
E:\conda_envs\football-dl\python.exe -m src.models.mlp
```

Redo this after regenerating the dataset (new feature, updated data, etc.).
Automatically saved to `models_saved/` (not versioned).

---

## Predict a match

Easiest way: edit [test_predict.py](test_predict.py) (at the root) then run:

```powershell
E:\conda_envs\football-dl\python.exe test_predict.py
```

Minimal example (model only, no manual adjustments):
```python
from src.models.predict import predict_match
predict_match("Arsenal", "Chelsea")
```

Full example (all available manual adjustments):
```python
predict_match(
    "Man City", "Sunderland",
    injured_home=["Rodri"],              # home players out
    injured_away=[],                     # away players out
    lineup_home=[...],                   # confirmed starting XI, 11 names (complements injured_home)
    lineup_away=[...],
    rest_days_diff=2,                    # rest-day gap (+ = home team more rested)
    home_position=1, home_points=68,     # CURRENT home standings
    away_position=17, away_points=31,    # CURRENT away standings
    stakes_home="title",                 # "title" / "europe" / "survival" / "neutral"
    stakes_away="survival",
    derby=False,                         # True/False
    odds_1x2={"1": 1.25, "X": 6.5, "2": 11.0},  # bookmaker odds
    match_date="2026-01-02",             # match date (default = today)
    show_markets=True,                   # probable scores / BTTS / over-under (default True)
)
```

Or in a single line, without going through the file:
```powershell
E:\conda_envs\football-dl\python.exe -c "from src.models.predict import predict_match; predict_match('Arsenal', 'Chelsea')"
```
(⚠ type this directly into the terminal, never paste multi-line Python code as-is
into PowerShell — it breaks: PowerShell tries to interpret it as PowerShell, not
Python)

---

## Update squads / market values

File: [src/features/squad_values.py](src/features/squad_values.py) — `SQUAD_VALUES`
dict, one block per club, `"Player Name": value_in_millions`. Direct manual edit,
nothing to retrain afterwards (only used by `predict_match()` to weight the impact
of injuries, never by training).

---

## Quick reference

| I want to... | Command |
|---|---|
| Update the dataset | `python -m src.features.build_dataset` |
| Check nothing is broken | `python -m pytest tests/ -v` |
| Retrain the baseline LR | `python -m src.models.baseline_lr` |
| Retrain the MLP | `python -m src.models.mlp` |
| Predict a match | edit `test_predict.py` then `python test_predict.py` |
| Change a squad/value | edit `src/features/squad_values.py` |

(replace `python` with `E:\conda_envs\football-dl\python.exe`)
