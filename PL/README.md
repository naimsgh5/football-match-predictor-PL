# Football Match Predictor — Premier League

Predicts the outcome of a Premier League match (home win / draw / away win), comparing several approaches: baseline Logistic Regression → MLP → LSTM/Transformer (PyTorch).

Personal project to build up deep learning skills, following on from an earlier project using Logistic Regression One-vs-All + Elo ratings + Monte Carlo.

> Common commands (updating data, training, prediction): see [COMMANDS.md](COMMANDS.md).
> **All commands below run from this folder** (`cd PL` from the repo root).

## Structure

```
data/
  raw/         # raw data, never modified
  interim/     # cleaned data (rebuildable, not versioned)
  processed/   # final training-ready features (rebuildable, not versioned)
notebooks/     # exploration (EDA, model comparison)
src/
  data/        # collection and cleaning
  features/    # Elo, form, head-to-head, standings, injuries
  models/      # baseline, MLP, LSTM/Transformer
  evaluation/  # metrics, calibration
tests/         # unit tests (notably anti-data-leakage)
configs/       # per-experiment hyperparameters
models_saved/  # checkpoints (not versioned)
```

## Setup

```bash
pip install -r requirements.txt
```

> Local environment: since `C:` is nearly full (very little free space), the project's
> conda environment lives on `E:\conda_envs\football-dl` instead of the default Anaconda
> install. `python -m ...` commands in this README assume
> `E:\conda_envs\football-dl\python.exe` is on PATH (or called explicitly).

## Milestones

- [x] **M0 — Repo setup**
  - Folder structure (`data/`, `src/`, `notebooks/`, `tests/`, `configs/`)
  - `.gitignore`, `requirements.txt`, README

- [x] **M1 — Raw Premier League data**
  - `data/raw/premier_league_results.csv`: 1900 matches, 5 full seasons (2021/22 → 2025/26, 380 matches each)
  - 2025/26 season completed via `data/raw/E0_2025_26_footballdata.csv` (football-data.co.uk): 21 missing matches added, 0 score discrepancies on matches already present
  - Validated in `notebooks/predictor.ipynb`: 0 missing values, home/away team names consistent

- [x] **M2 — Feature engineering**
  - Modules in `src/features/`: pre-match Elo, rolling form (last 10 matches), rolling goals scored/conceded, head-to-head, average rank over the last 5 completed seasons
  - Injuries/market value utility (`market_value_injuries.py`) — reserved for inference on an upcoming match, not usable as a training feature (no injury history available)
  - Anti-lookahead-leakage tests (`tests/test_features.py`): 2 leakage bugs found and fixed (unstable date sort, fallback goals average computed on the whole dataset instead of the history known so far)
  - Final dataset: `data/processed/premier_league_features.parquet`, 1900 matches × 10 features (`elo_diff`, `form_diff`, `venue_form_diff`, `h2h_home_win_rate`, `attack_diff`, `defense_diff`, `rank_diff`, `congestion_diff`, `quality_form_diff`, `clean_sheet_diff`)
  - `venue_form_diff`: form computed separately for home / away (added later, see M3)
  - `congestion_diff` (added later, see M4): proxy for fixture congestion / squad-rotation risk — number of matches each team played in the preceding 10 days. Known limitation: only counts PL matches from this dataset, not cup/European matches (source not available) — underestimates the true fatigue of a team competing on multiple fronts. Non-zero on 13.8% of matches (festive period, midweek rounds)

- [x] **M3 — Baseline Logistic Regression**
  - `src/models/baseline_lr.py`: sklearn multinomial `LogisticRegression`, standardized features (`StandardScaler`)
  - Strict temporal split: train = 2021-2023 seasons (1140 matches), validation = 2024 season, test = 2025 season (never random shuffle)
  - Results: accuracy 51.6% (val) / 48.2% (test), vs 40.8% / 42.6% for the naive baseline ("always home")
  - Known limitation: the model almost never predicts a draw (the hardest class to separate) — worth watching on later models
  - `src/evaluation/metrics.py`: evaluation functions reused for M4/M5 (accuracy, log-loss, confusion matrix)
  - `src/models/predict.py`: predicts a specific upcoming match (`python -m src.models.predict "Arsenal" "Chelsea"`), recomputes automatic features from the final state of the history; optional post-hoc adjustments, entered by hand (never learned by the model): injuries/market value (`src/features/squad_values.py`), rest days, current standings/points (current season, absent from the dataset), match stakes (title/europe/relegation/derby), bookmaker odds (market average, margin removed)
  - Retrained after adding `venue_form_diff`: accuracy 52.1%/47.9% (nearly identical — `venue_form_diff` has little importance for a linear model, worth watching on MLP/LSTM)
  - `src/features/squad_values.py`: market values refreshed from transfermarkt.co.uk (full squads, 20 clubs)
  - `src/models/markets.py`: probable exact scores / BTTS / over-under, via a **second model** (Poisson on expected goals, independent of the 1X2 classifier) — the 1X2 model only has 3 classes by construction, it can't give a score; shown automatically by `predict_match()` (`show_markets=True` by default). This goal model's implied 1X2 is shown next to the main model's for comparison, without being forced to match — two independent estimates. `tests/test_markets.py`: probability distribution consistency (sum to 1)
  - `test_predict.py` (root): ready-to-use sandbox for testing predictions by hand

- [x] **M4 — MLP (PyTorch)**
  - `src/models/mlp.py`: simple dense network (2 hidden layers 32/16, dropout 0.3, Adam), same features/split as M3
  - Full-batch training (the train set fits in a single batch given its small size) with early stopping on validation log-loss
  - Results: accuracy 53.4% (val) / 49.5% (test), log-loss 0.982 / 1.033 — slight improvement across the board over the LR baseline (52.1% / 47.9%, log-loss 0.996 / 1.040)
  - Same limitation as M3: draws are still ignored (0% recall) on both models — weak signal in the current features rather than a model-capacity limit
  - Retrained after adding `congestion_diff`: LR 51.6%/47.6%, MLP 51.6%/47.6% (slight dip, likely noise given the dataset size) — modest LR coefficient (0.042, 6th of 8) but not negligible, feature kept for M5 and computed automatically in `predict.py` (`match_date`, defaults to today)
  - `quality_form_diff` (10 features total): form weighted by opponent strength — rolling average of (result − expected result, from the Elo formula), so beating a strong team counts for more than beating a weak one, and losing to a weak team hurts more than losing to a strong one. Complements `form_diff` (which treats every result identically regardless of opponent) and `elo_diff` (which only partially captures this at the rating level, not the recent-form level)
  - `clean_sheet_diff`: rolling clean sheet rate (last 10 matches), distinct from `defense_diff` (average goals conceded can hide very different defensive consistency — e.g. `[0,0,0,0,3]` and `[1,1,1,1,1]` average the same 0.6 but have wildly different clean sheet counts)
  - Retrained after both: LR 51.8%/47.4%, MLP 51.3%/47.6% (roughly flat, within noise) — `quality_form_diff` lands 3rd by LR coefficient (0.185, above `attack_diff`), `clean_sheet_diff` 5th (0.100) — both meaningfully used by the model, not negligible
  - `predict.py`: `lineup_home`/`lineup_away` (confirmed starting XI, 11 names) — compares the fielded XI's value to the team's strongest possible XI, **complementing** `injured_home`/`injured_away` rather than replacing it (an injured/absent player and a fit-but-benched player are different signals, both tracked); realistically only usable ~1h before kickoff when lineups are confirmed
  - `predict.py` output now shows **both models' predictions separately** (clearly labelled "Model 1/2" / "Model 2/2"), instead of a single model — same post-hoc adjustments applied independently to each, so their reactions to the same inputs can be compared directly
  - Removed `markets.py::betting_notes()` (the "informational, not financial advice" text summary) — the Poisson goal markets (scores/BTTS/over-under) themselves are unchanged

- [ ] M5 — LSTM/Transformer (PyTorch)
- [ ] M6 — Final evaluation and model comparison
