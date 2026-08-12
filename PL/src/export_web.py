"""Exports trained model weights + historical state + squad values to a single JSON
file, for the web UI (Claude Artifact) that reimplements predict.py's logic in
JavaScript so predictions can run client-side, in the browser, without a Python backend.

The export is a frozen snapshot: whenever the dataset is updated or the models are
retrained, rerun this script and redeploy the artifact for it to reflect the new state.

Usage: python -m src.export_web
"""
import json
import os

import joblib
import numpy as np
import torch
from torch import nn

from src.features.build_dataset import FEATURE_COLUMNS, build_dataset_with_state
from src.features.rolling_stats import CLEAN_SHEET_DEFAULT_RATE, WINDOW
from src.features.head_to_head import WINDOW as H2H_WINDOW
from src.features.squad_values import SQUAD_VALUES
from src.models.mlp import MLP
from src.models.mlp import MODEL_PATH as MLP_MODEL_PATH
from src.models.mlp import SCALER_PATH as MLP_SCALER_PATH
from src.models.predict import LR_MODEL_PATH, LR_SCALER_PATH

OUT_PATH = "data/processed/web_export.json"
MATCH_DATES_KEEP = 15  # only the most recent dates matter for the 10-day congestion window


def _trim(history: dict, n: int) -> dict:
    return {team: list(vals[-n:]) for team, vals in history.items()}


def _export_mlp_layers(mlp_model: MLP) -> list:
    """Extracts (weight, bias) for each nn.Linear in the Sequential, in forward order.
    PyTorch Linear weight shape is (out_features, in_features): y = x @ W.T + b."""
    linear_layers = [m for m in mlp_model.net if isinstance(m, nn.Linear)]
    return [
        {"weight": layer.weight.detach().numpy().tolist(), "bias": layer.bias.detach().numpy().tolist()}
        for layer in linear_layers
    ]


def main():
    _df, state = build_dataset_with_state()

    lr_model = joblib.load(LR_MODEL_PATH)
    lr_scaler = joblib.load(LR_SCALER_PATH)

    mlp_model = MLP(input_dim=len(FEATURE_COLUMNS))
    mlp_model.load_state_dict(torch.load(MLP_MODEL_PATH))
    mlp_model.eval()
    mlp_scaler = joblib.load(MLP_SCALER_PATH)

    teams = sorted(SQUAD_VALUES.keys())

    h2h_export = {
        f"{a}|{b}": list(vals[-H2H_WINDOW:])
        for (a, b), vals in state["h2h_history"].items()
    }

    match_dates_export = {
        team: [d.strftime("%Y-%m-%d") for d in sorted(dates)[-MATCH_DATES_KEEP:]]
        for team, dates in state["match_dates"].items()
    }

    standings_export = {
        str(season): ranks for season, ranks in state["standings"].items()
    }

    export = {
        "teams": teams,
        "feature_columns": FEATURE_COLUMNS,
        "state": {
            "elo": state["elo"],
            "form_history": _trim(state["form_history"], WINDOW),
            "quality_form_history": _trim(state["quality_form_history"], WINDOW),
            "home_venue_history": _trim(state["home_venue_history"], WINDOW),
            "away_venue_history": _trim(state["away_venue_history"], WINDOW),
            "scored": _trim(state["scored"], WINDOW),
            "conceded": _trim(state["conceded"], WINDOW),
            "clean_sheet_history": _trim(state["clean_sheet_history"], WINDOW),
            "h2h_history": h2h_export,
            "standings": standings_export,
            "match_dates": match_dates_export,
        },
        "lr_model": {
            "classes": lr_model.classes_.tolist(),
            "coef": lr_model.coef_.tolist(),
            "intercept": lr_model.intercept_.tolist(),
            "scaler_mean": lr_scaler.mean_.tolist(),
            "scaler_scale": lr_scaler.scale_.tolist(),
        },
        "mlp_model": {
            "scaler_mean": mlp_scaler.mean_.tolist(),
            "scaler_scale": mlp_scaler.scale_.tolist(),
            "layers": _export_mlp_layers(mlp_model),
        },
        "squad_values": SQUAD_VALUES,
        "defaults": {
            "clean_sheet_default_rate": CLEAN_SHEET_DEFAULT_RATE,
        },
    }

    os.makedirs("data/processed", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(export, f, separators=(",", ":"))

    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"{len(teams)} teams, {len(state['elo'])} teams with elo history -> {OUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
