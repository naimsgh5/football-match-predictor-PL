# Football Match Predictor (Deep Learning)

Prédiction du résultat d'un match de Premier League (victoire domicile / nul / victoire extérieur), en comparant plusieurs approches : baseline Logistic Regression → MLP → LSTM/Transformer (PyTorch).

Projet personnel de montée en compétences en deep learning, dans la continuité d'un premier projet en Logistic Regression One-vs-All + Elo ratings + Monte Carlo (voir [`algo/`](algo/) pour l'ancienne version, gardée comme référence).

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

- [x] M0 — Setup du repo
- [ ] M1 — Données brutes Premier League (validation, nettoyage)
- [ ] M2 — Feature engineering (Elo, forme, head-to-head, classement 5 ans, blessures/valeur marchande)
- [ ] M3 — Baseline Logistic Regression
- [ ] M4 — MLP (PyTorch)
- [ ] M5 — LSTM/Transformer (PyTorch)
- [ ] M6 — Évaluation finale et comparaison des modèles
