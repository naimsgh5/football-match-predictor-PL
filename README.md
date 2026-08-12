# Football Match Predictor (Deep Learning)

Prédiction de résultats de matchs de football via plusieurs approches de deep learning
(Logistic Regression → MLP → LSTM/Transformer), organisée **par compétition** — chaque
championnat a son propre pipeline (données, features, modèles), pour rester comparable en
interne (voir la discussion dans `PL/README.md` sur pourquoi mélanger des championnats
différents dans un même Elo/classement n'a pas de sens sans recalibration).

## Compétitions

- **[PL/](PL/)** — Premier League. La seule active pour l'instant ; voir
  [PL/README.md](PL/README.md) pour le détail des milestones, et [PL/COMMANDS.md](PL/COMMANDS.md)
  pour les commandes courantes (mise à jour des données, entraînement, prédiction).
- D'autres championnats (LaLiga, Bundesliga, Ligue 1, Serie A) et un module **LDC**
  (Ligue des Champions, qui assemblera des infos venant de plusieurs championnats -- forme,
  fatigue, blessures directement réutilisables, mais Elo/classement à recalibrer entre
  championnats, ex. via les coefficients UEFA) sont prévus mais pas encore commencés.

## Structure du repo

```
PL/            # Premier League -- pipeline complet (voir PL/README.md)
algo/          # ancien projet (reference locale, hors repo -- gitignore)
```

Chaque dossier de compétition est censé rester autonome (ses propres `data/`, `src/`,
`tests/`, `models_saved/`) mais partager le même code générique dès que possible plutôt que
de le dupliquer -- à organiser en `src/` commun si/quand un deuxième championnat démarre.
