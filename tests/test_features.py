"""Tests anti-fuite temporelle : les features d'un match ne doivent jamais dépendre
de matchs qui n'ont pas encore eu lieu à sa date.

Méthode : on construit le dataset sur l'historique complet, puis sur une version tronquée
(uniquement les matchs jusqu'à une date de coupure). Les features des matchs présents dans
les deux versions doivent être strictement identiques — sinon une feature "voit" le futur.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.build_dataset import FEATURE_COLUMNS, RAW_PATH, build_dataset

TRUNCATED_PATH = "data/interim/_test_truncated.csv"


@pytest.fixture(scope="module")
def full_dataset():
    return build_dataset(RAW_PATH)


def test_no_lookahead_leakage(full_dataset):
    cutoff = full_dataset["date"].iloc[len(full_dataset) // 2]

    raw = pd.read_csv(RAW_PATH, parse_dates=["date"]).sort_values("date", kind="stable")
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    raw[raw["date"] <= cutoff].to_csv(TRUNCATED_PATH, index=False)

    try:
        df_truncated = build_dataset(TRUNCATED_PATH)
        df_full_prefix = full_dataset[full_dataset["date"] <= cutoff].reset_index(drop=True)

        assert len(df_truncated) == len(df_full_prefix)
        for col in FEATURE_COLUMNS:
            assert np.allclose(df_full_prefix[col].values, df_truncated[col].values), (
                f"Fuite de donnees detectee sur la colonne '{col}'"
            )
    finally:
        Path(TRUNCATED_PATH).unlink(missing_ok=True)


def test_first_season_has_no_rank_history(full_dataset):
    first_season = full_dataset["season"].min()
    rows = full_dataset[full_dataset["season"] == first_season]
    assert (rows["rank_diff"] == 0).all(), (
        "La premiere saison du dataset ne doit avoir aucun historique de classement (rank_diff=0 attendu)"
    )


def test_elo_starts_at_initial_rating_for_new_teams(full_dataset):
    first_match = full_dataset.iloc[0]
    assert first_match["elo_home"] == 1500
    assert first_match["elo_away"] == 1500
