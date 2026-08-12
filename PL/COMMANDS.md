# Commandes utiles

Aide-mémoire des commandes courantes du projet Premier League. **Tout se lance depuis ce
dossier** (`e:\PROJET DL\PL`) :

```powershell
cd "e:\PROJET DL\PL"
```

Avec l'interpréteur Python de l'environnement conda sur `E:` (le disque `C:` est saturé, voir README) :

```
E:\conda_envs\football-dl\python.exe
```

Pour éviter de le retaper à chaque fois, tu peux l'ajouter temporairement à ta session
PowerShell :
```powershell
$py = "E:\conda_envs\football-dl\python.exe"
```
puis utiliser `& $py ...` dans les commandes ci-dessous (sinon, remplace `python` par le
chemin complet à chaque fois).

---

## Mettre à jour les données

Le dataset (`data/raw/premier_league_results.csv`) est mis à jour **à la main** (pas
d'automatisation pour l'instant) :

1. Retélécharger les derniers résultats Premier League (ex. depuis football-data.co.uk,
   même source que `data/raw/E0_2025_26_footballdata.csv`)
2. Fusionner les nouveaux matchs dans `data/raw/premier_league_results.csv` (même méthode
   qu'en M1 : pas de doublon, pas de divergence de score sur les matchs déjà présents)
3. Régénérer le dataset de features :

```powershell
E:\conda_envs\football-dl\python.exe -m src.features.build_dataset
```

Fréquence conseillée : une fois par semaine en saison (les équipes jouent ~1x/semaine),
et surtout **juste avant** de prédire un match précis, pour que l'historique soit à jour.

---

## Vérifier que tout fonctionne (tests)

```powershell
E:\conda_envs\football-dl\python.exe -m pytest tests/ -v
```

À relancer après toute modification des features ou du dataset (anti-fuite temporelle,
cohérence des marchés Poisson, etc.).

---

## Réentraîner les modèles

```powershell
# Baseline Logistic Regression (M3)
E:\conda_envs\football-dl\python.exe -m src.models.baseline_lr

# MLP PyTorch (M4)
E:\conda_envs\football-dl\python.exe -m src.models.mlp
```

À refaire après avoir régénéré le dataset (nouvelle feature, données mises à jour, etc.).
Sauvegarde automatique dans `models_saved/` (non versionné).

---

## Prédire un match

Le plus simple : éditer [test_predict.py](test_predict.py) (à la racine) puis lancer :

```powershell
E:\conda_envs\football-dl\python.exe test_predict.py
```

Exemple minimal (juste le modèle, aucun ajustement manuel) :
```python
from src.models.predict import predict_match
predict_match("Arsenal", "Chelsea")
```

Exemple complet (tous les ajustements manuels disponibles) :
```python
predict_match(
    "Man City", "Sunderland",
    injured_home=["Rodri"],              # joueurs absents domicile
    injured_away=[],                     # joueurs absents exterieur
    rest_days_diff=2,                    # ecart de jours de repos (+ = domicile plus repose)
    home_position=1, home_points=68,     # classement ACTUEL domicile
    away_position=17, away_points=31,    # classement ACTUEL exterieur
    stakes_home="titre",                 # "titre" / "europe" / "maintien" / "neutre"
    stakes_away="maintien",
    derby=False,                         # True/False
    odds_1x2={"1": 1.25, "X": 6.5, "2": 11.0},  # cotes bookmaker
    match_date="2026-01-02",             # date du match (defaut = aujourd'hui)
    show_markets=True,                   # scores probables / BTTS / over-under (defaut True)
)
```

Ou en une ligne, sans passer par le fichier :
```powershell
E:\conda_envs\football-dl\python.exe -c "from src.models.predict import predict_match; predict_match('Arsenal', 'Chelsea')"
```
(⚠ à taper directement dans le terminal, jamais coller du code Python multi-lignes tel quel
dans PowerShell — ça casse : PowerShell essaie de l'interpréter comme du PowerShell, pas
du Python)

---

## Mettre à jour les effectifs / valeurs marchandes

Fichier : [src/features/squad_values.py](src/features/squad_values.py) — dict `SQUAD_VALUES`,
un bloc par club, `"Nom Joueur": valeur_en_millions`. Édition manuelle directe, rien à
réentraîner ensuite (utilisé uniquement par `predict_match()` pour pondérer l'impact des
blessures, jamais par l'entraînement).

---

## Récapitulatif express

| Je veux... | Commande |
|---|---|
| Mettre à jour le dataset | `python -m src.features.build_dataset` |
| Vérifier que rien n'est cassé | `python -m pytest tests/ -v` |
| Réentraîner la baseline LR | `python -m src.models.baseline_lr` |
| Réentraîner le MLP | `python -m src.models.mlp` |
| Prédire un match | éditer `test_predict.py` puis `python test_predict.py` |
| Changer un effectif/valeur | éditer `src/features/squad_values.py` |

(remplacer `python` par `E:\conda_envs\football-dl\python.exe`)
