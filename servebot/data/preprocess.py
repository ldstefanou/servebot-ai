from typing import Optional

import numpy as np
import pandas as pd
from paths import get_dataset_path


def walk_back_split(df, target_validation_players=100):
    validation_players = set()
    validation_cutoff_idx = len(df)

    # Walk backwards through the sorted dataframe
    for idx, row in df[::-1].iterrows():
        # Check if we'd add any new players
        new_players = {row["winner_name"], row["loser_name"]} - validation_players

        # Add the new players
        validation_players.update(new_players)

        # Stop when we have enough unique validation players
        if len(validation_players) >= target_validation_players:
            validation_cutoff_idx = idx
            break

    # Add validation flag
    df["is_validation"] = df.index >= validation_cutoff_idx

    return df


def sample_df_by_match_id(df: pd.DataFrame, size: int):
    unique_matches = df["match_id"].unique()
    sampled_matches = np.random.choice(
        unique_matches, size=min(size, len(unique_matches)), replace=False
    )
    df = df[df["match_id"].isin(sampled_matches)]
    return df


def load_training_dataframe():
    df = pd.read_parquet(get_dataset_path("all"))
    df["date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")
    df["time_since_epoch"] = df["date"].astype("int64") // 10**9
    df = df.sort_values(["date", "tourney_id", "match_num"])
    return df


def bin_categorical_features(df: pd.DataFrame):
    """Create categorical bins for age, height, and rank features"""

    # Age binning - career stage bins
    df["winner_age"] = df["winner_age"].fillna(df["winner_age"].mean())
    df["loser_age"] = df["loser_age"].fillna(df["loser_age"].mean())

    age_bins = [0, 18, 21, 25, 29, 33, float("inf")]
    age_labels = [
        "Junior",
        "Rising",
        "Prime Early",
        "Prime Peak",
        "Veteran",
        "Late Career",
    ]

    df["winner_age_group"] = pd.cut(
        df["winner_age"], bins=age_bins, labels=age_labels
    ).astype(str)
    df["loser_age_group"] = pd.cut(
        df["loser_age"], bins=age_bins, labels=age_labels
    ).astype(str)

    # Height binning - clean outliers and create strategic bins
    df["winner_ht"] = df["winner_ht"].replace(0, np.nan)  # Replace 0s with NaN
    df["loser_ht"] = df["loser_ht"].replace(0, np.nan)

    # Filter out clearly wrong heights (< 150cm or > 220cm)
    df.loc[df["winner_ht"] < 150, "winner_ht"] = np.nan
    df.loc[df["winner_ht"] > 220, "winner_ht"] = np.nan
    df.loc[df["loser_ht"] < 150, "loser_ht"] = np.nan
    df.loc[df["loser_ht"] > 220, "loser_ht"] = np.nan

    # Fill missing heights with mean
    df["winner_ht"] = df["winner_ht"].fillna(df["winner_ht"].mean())
    df["loser_ht"] = df["loser_ht"].fillna(df["loser_ht"].mean())

    height_bins = [0, 175, 185, 193, float("inf")]
    height_labels = ["Short", "Average", "Tall", "Very Tall"]

    df["winner_height_group"] = pd.cut(
        df["winner_ht"], bins=height_bins, labels=height_labels
    ).astype(str)
    df["loser_height_group"] = pd.cut(
        df["loser_ht"], bins=height_bins, labels=height_labels
    ).astype(str)

    # Rank binning
    ranking_bins = [-1, 0, 1, 5, 10, 20, 50, 100, 200, 500, 1000, float("inf")]
    ranking_labels = [
        "Unranked",
        "#1",
        "Top 5",
        "Top 10",
        "Top 20",
        "Top 50",
        "Top 100",
        "Top 200",
        "Top 500",
        "Top 1000",
        "1000+",
    ]

    df["winner_rank"] = pd.cut(
        df["winner_rank"].fillna(0), bins=ranking_bins, labels=ranking_labels
    ).astype(str)
    df["loser_rank"] = pd.cut(
        df["loser_rank"].fillna(0), bins=ranking_bins, labels=ranking_labels
    ).astype(str)

    return df


def clean_dataframe(df: pd.DataFrame):
    # time stuff
    df["year"] = df["date"].dt.year

    # h2h scores
    sets = df[["winner_name", "loser_name"]].apply(frozenset, axis=1)
    df["h2h_winner_player"] = df.groupby([sets, "winner_name"]).cumcount()
    df["h2h_loser_player"] = df.groupby(sets).cumcount() - df["h2h_winner_player"]

    # match id
    df["match_id"] = np.arange(df.shape[0])

    # Apply categorical binning
    df = bin_categorical_features(df)

    df["year"] = df["date"].dt.year
    df["tourney_str"] = (
        df["tourney_name"].str.split(" ").apply(lambda x: " ".join(x[:4]))
    )
    biggest_tourney = (
        df["tourney_str"].value_counts().sort_values(ascending=False)[:100]
    )
    df["tournament"] = np.where(
        df["tourney_str"].isin(biggest_tourney.index),
        df["tourney_str"],
        "Other tournament",
    )
    return df


def set_validation_matches(
    df: pd.DataFrame, validate_on_last_slam: bool = True, validation_size=0.05
):

    if validate_on_last_slam:
        # Define Grand Slam names
        grand_slams = ["australian open", "roland garros", "wimbledon", "us open"]
        is_slam = df["tourney_str"].str.lower().str.contains("|".join(grand_slams))
        tourney_id = df.groupby(is_slam)["tourney_id"].last().loc[True]

        # Mark validation rows
        df["is_validation"] = df["tourney_id"].eq(tourney_id)
        last_slam_index = df[df["tourney_id"] == tourney_id].index[-1]

        # Mark validation rows (this marks the last Grand Slam matches)
        df["is_validation"] = df["tourney_id"].eq(tourney_id)

        # Now filter out rows that come after the last index of the last Grand Slam
        df = df.loc[:last_slam_index].copy()

    else:
        val_size = int(len(df) * (1 - validation_size))
        validation_match_ids = df[val_size:]["match_id"].unique()
        df["is_validation"] = df["match_id"].isin(validation_match_ids)

    return df


def load_data(sample: Optional[int] = None, filter_matches: str = "all"):
    df = load_training_dataframe()
    df = clean_dataframe(df)

    # Filter by match type
    if filter_matches == "atp":
        df = df[df["source"] == "atp"]

    if sample is not None:
        df = sample_df_by_match_id(df, sample)
    return df
