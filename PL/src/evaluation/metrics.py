"""Evaluation metrics shared across all models (baseline, MLP, LSTM)."""
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

RESULT_LABELS = {0: "Away", 1: "Draw", 2: "Home"}
LABEL_ORDER = [0, 1, 2]


def evaluate(y_true, y_pred, y_proba, label: str = "") -> dict:
    """Prints accuracy, log-loss, confusion matrix and classification report.
    Returns a dict {accuracy, log_loss} reusable to compare models against each other."""
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_proba, labels=LABEL_ORDER)

    names = [RESULT_LABELS[i] for i in LABEL_ORDER]

    if label:
        print(f"--- {label} ---")
    print(f"Accuracy : {acc:.3f}")
    print(f"Log-loss : {ll:.3f}")
    print()
    print("Confusion matrix (rows = actual, columns = predicted):")
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    print(pd.DataFrame(cm, index=names, columns=names))
    print()
    print(classification_report(y_true, y_pred, labels=LABEL_ORDER, target_names=names))

    return {"accuracy": acc, "log_loss": ll}
