"""Baseline Logistic Regression (multinomiale, sklearn) sur les features de M2.

Split temporel strict — jamais de shuffle aléatoire : le modèle est entraîné sur les
saisons les plus anciennes et évalué sur les plus récentes, jamais l'inverse.

Usage : python -m src.models.baseline_lr
"""
import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import RESULT_LABELS, evaluate
from src.features.build_dataset import FEATURE_COLUMNS, OUT_PATH

VAL_SEASON = 2024
TEST_SEASON = 2025  # train = toutes les saisons strictement avant VAL_SEASON

MODEL_PATH = "models_saved/baseline_lr.joblib"
SCALER_PATH = "models_saved/baseline_lr_scaler.joblib"


def load_splits(path: str = OUT_PATH):
    df = pd.read_parquet(path)
    train = df[df["season"] < VAL_SEASON]
    val = df[df["season"] == VAL_SEASON]
    test = df[df["season"] == TEST_SEASON]
    return train, val, test


def to_xy(split: pd.DataFrame):
    return split[FEATURE_COLUMNS].values, split["result"].values


def main():
    train, val, test = load_splits()
    print(f"Train : {len(train)} matchs (saisons < {VAL_SEASON})")
    print(f"Val   : {len(val)} matchs (saison {VAL_SEASON})")
    print(f"Test  : {len(test)} matchs (saison {TEST_SEASON})")
    print()

    X_train, y_train = to_xy(train)
    X_val, y_val = to_xy(val)
    X_test, y_test = to_xy(test)

    scaler = StandardScaler().fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_sc, y_train)

    majority_class = pd.Series(y_train).mode()[0]
    naive_val_acc = (y_val == majority_class).mean()
    naive_test_acc = (y_test == majority_class).mean()
    print(f"Baseline naive (toujours predire '{RESULT_LABELS[majority_class]}') "
          f"- Val: {naive_val_acc:.3f}  Test: {naive_test_acc:.3f}")
    print()

    y_val_pred = model.predict(X_val_sc)
    y_val_proba = model.predict_proba(X_val_sc)
    evaluate(y_val, y_val_pred, y_val_proba, label=f"VALIDATION (saison {VAL_SEASON})")

    y_test_pred = model.predict(X_test_sc)
    y_test_proba = model.predict_proba(X_test_sc)
    evaluate(y_test, y_test_pred, y_test_proba, label=f"TEST (saison {TEST_SEASON})")

    print("Coefficients (feature -> poids moyen |classe|) :")
    coef_importance = pd.Series(
        abs(model.coef_).mean(axis=0), index=FEATURE_COLUMNS
    ).sort_values(ascending=False)
    print(coef_importance)

    os.makedirs("models_saved", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nModele sauvegarde -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
