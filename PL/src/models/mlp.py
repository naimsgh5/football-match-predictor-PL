"""MLP (PyTorch) sur les mêmes features et le même split temporel que la baseline LR (M3).

Petit réseau (2 couches cachées, dropout) volontairement simple : le train set ne fait que
~1140 lignes pour 7 features, un modèle trop capacitaire overfitterait instantanément.
Entraînement full-batch (tout le train tient en un seul batch) avec early stopping sur le
log-loss de validation.

Usage : python -m src.models.mlp
"""
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.evaluation.metrics import RESULT_LABELS, evaluate
from src.features.build_dataset import FEATURE_COLUMNS, OUT_PATH

VAL_SEASON = 2024
TEST_SEASON = 2025  # train = toutes les saisons strictement avant VAL_SEASON

HIDDEN_SIZES = (32, 16)
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 1000
PATIENCE = 50  # early stopping sur le log-loss de validation
SEED = 42

MODEL_PATH = "models_saved/mlp.pt"
SCALER_PATH = "models_saved/mlp_scaler.joblib"


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes=HIDDEN_SIZES, dropout: float = DROPOUT, n_classes: int = 3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_splits(path: str = OUT_PATH):
    df = pd.read_parquet(path)
    train = df[df["season"] < VAL_SEASON]
    val = df[df["season"] == VAL_SEASON]
    test = df[df["season"] == TEST_SEASON]
    return train, val, test


def to_xy(split: pd.DataFrame):
    X = split[FEATURE_COLUMNS].values.astype(np.float32)
    y = split["result"].values.astype(np.int64)
    return X, y


def train_model(X_train, y_train, X_val, y_val, input_dim: int):
    torch.manual_seed(SEED)
    model = MLP(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping a l'epoch {epoch + 1} (meilleur val_loss={best_val_loss:.4f})")
                break
    else:
        print(f"MAX_EPOCHS atteint (meilleur val_loss={best_val_loss:.4f})")

    model.load_state_dict(best_state)
    return model, history


def predict_proba(model: MLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype(np.float32)))
        proba = torch.softmax(logits, dim=1).numpy()
    return proba


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
    X_train_sc = scaler.transform(X_train).astype(np.float32)
    X_val_sc = scaler.transform(X_val).astype(np.float32)
    X_test_sc = scaler.transform(X_test).astype(np.float32)

    majority_class = pd.Series(y_train).mode()[0]
    naive_val_acc = (y_val == majority_class).mean()
    naive_test_acc = (y_test == majority_class).mean()
    print(f"Baseline naive (toujours predire '{RESULT_LABELS[majority_class]}') "
          f"- Val: {naive_val_acc:.3f}  Test: {naive_test_acc:.3f}")
    print()

    model, history = train_model(X_train_sc, y_train, X_val_sc, y_val, input_dim=len(FEATURE_COLUMNS))
    print()

    y_val_proba = predict_proba(model, X_val_sc)
    y_val_pred = y_val_proba.argmax(axis=1)
    evaluate(y_val, y_val_pred, y_val_proba, label=f"VALIDATION (saison {VAL_SEASON})")

    y_test_proba = predict_proba(model, X_test_sc)
    y_test_pred = y_test_proba.argmax(axis=1)
    evaluate(y_test, y_test_pred, y_test_proba, label=f"TEST (saison {TEST_SEASON})")

    os.makedirs("models_saved", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Modele sauvegarde -> {MODEL_PATH}")

    return history


if __name__ == "__main__":
    main()
