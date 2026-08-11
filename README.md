# Football Match Predictor (Deep Learning)

Prédiction du résultat d'un match de Premier League (victoire domicile / nul / victoire extérieur), en comparant plusieurs approches : baseline Logistic Regression → MLP → LSTM/Transformer (PyTorch).

Projet personnel de montée en compétences en deep learning, dans la continuité d'un premier projet en Logistic Regression One-vs-All + Elo ratings + Monte Carlo.

## Structure

```
data/
  raw/         # données brutes, jamais modifiées
  interim/     # données nettoyées (régénérable, non versionné)
  processed/   # features finales prêtes pour l'entraînement (régénérable, non versionné)
notebooks/     # exploration (EDA, comparaison de modèles)
src/
  data/        # collecte et nettoyage
  features/    # Elo, forme, head-to-head, classement, blessures
  models/      # baseline, MLP, LSTM/Transformer
  evaluation/  # métriques, calibration
tests/         # tests unitaires (notamment anti-data-leakage)
configs/       # hyperparamètres par expérience
models_saved/  # checkpoints (non versionné)
```

## Setup

```bash
pip install -r requirements.txt
```

## Milestones

- [x] **M0 — Setup du repo**
  - Structure de dossiers (`data/`, `src/`, `notebooks/`, `tests/`, `configs/`)
  - `.gitignore`, `requirements.txt`, README

- [x] **M1 — Données brutes Premier League**
  - `data/raw/premier_league_results.csv` : 1900 matchs, 5 saisons complètes (2021/22 → 2025/26, 380 matchs chacune)
  - Saison 2025/26 complétée via `data/raw/E0_2025_26_footballdata.csv` (football-data.co.uk) : 21 matchs manquants ajoutés, 0 divergence de score sur les matchs déjà présents
  - Validation dans `notebooks/predictor.ipynb` : 0 valeur manquante, noms d'équipes cohérents domicile/extérieur

- [x] **M2 — Feature engineering**
  - Modules dans `src/features/` : Elo pré-match, forme glissante (10 derniers matchs), buts marqués/encaissés glissants, head-to-head, classement moyen des 5 dernières saisons complètes
  - Utilitaire blessures/valeur marchande (`market_value_injuries.py`) — réservé à l'inférence sur un match à venir, pas utilisable comme feature d'entraînement (pas d'historique de blessures disponible)
  - Tests anti-fuite temporelle (`tests/test_features.py`) : 2 bugs de fuite détectés et corrigés (tri de date non stable, moyenne de buts de repli calculée sur le dataset entier au lieu de l'historique déjà connu)
  - Dataset final : `data/processed/premier_league_features.parquet`, 1900 matchs × 7 features (`elo_diff`, `form_diff`, `venue_form_diff`, `h2h_home_win_rate`, `attack_diff`, `defense_diff`, `rank_diff`)
  - `venue_form_diff` : forme calculée séparément à domicile / à l'extérieur (ajoutée après coup, cf M3)

- [x] **M3 — Baseline Logistic Regression**
  - `src/models/baseline_lr.py` : sklearn `LogisticRegression` multinomiale, features standardisées (`StandardScaler`)
  - Split temporel strict : train = saisons 2021-2023 (1140 matchs), validation = saison 2024, test = saison 2025 (jamais de shuffle aléatoire)
  - Résultats : accuracy 51.6% (val) / 48.2% (test), contre 40.8% / 42.6% pour la baseline naïve ("toujours domicile")
  - Limite connue : le modèle prédit quasiment jamais le nul (classe la plus difficile à séparer) — à surveiller sur les modèles suivants
  - `src/evaluation/metrics.py` : fonctions d'évaluation réutilisées pour M4/M5 (accuracy, log-loss, matrice de confusion)
  - `src/models/predict.py` : prédiction d'un match précis à venir (`python -m src.models.predict "Arsenal" "Chelsea"`), recalcule les features automatiques à partir de l'état final de l'historique ; ajustements optionnels post-hoc, saisis à la main (jamais appris par le modèle) : blessures/valeur marchande (`src/features/squad_values.py`), jours de repos, classement/points actuels (saison en cours, absente du dataset), enjeu du match (titre/europe/maintien/derby), cotes bookmaker (moyenne marché, marge retirée)
  - Réentraîné après l'ajout de `venue_form_diff` : accuracy 52.1%/47.9% (quasi identique, `venue_form_diff` a peu d'importance pour un modèle linéaire — à surveiller sur MLP/LSTM)
- [ ] M4 — MLP (PyTorch)
- [ ] M5 — LSTM/Transformer (PyTorch)
- [ ] M6 — Évaluation finale et comparaison des modèles
