"""Anti-lookahead-leakage tests: a match's features must never depend on matches that
haven't happened yet as of its date.

Method: build the dataset on the full history, then on a truncated version (only matches
up to a cutoff date). The features of matches present in both versions must be strictly
identical — otherwise a feature is "seeing" the future.
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
                f"Data leakage detected in column '{col}'"
            )
    finally:
        Path(TRUNCATED_PATH).unlink(missing_ok=True)


def test_first_season_has_no_rank_history(full_dataset):
    first_season = full_dataset["season"].min()
    rows = full_dataset[full_dataset["season"] == first_season]
    assert (rows["rank_diff"] == 0).all(), (
        "The dataset's first season should have no rank history (rank_diff=0 expected)"
    )


def test_elo_starts_at_initial_rating_for_new_teams(full_dataset):
    first_match = full_dataset.iloc[0]
    assert first_match["elo_home"] == 1500
    assert first_match["elo_away"] == 1500


def test_congestion_is_zero_without_recent_history(full_dataset):
    first_match = full_dataset.iloc[0]
    assert first_match["congestion_diff"] == 0, (
        "The dataset's very first match should have no recent history -> congestion_diff=0 expected"
    )
