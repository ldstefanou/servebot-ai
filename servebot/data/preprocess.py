from typing import Optional

import numpy as np
import pandas as pd


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


def prepare_df(
    df: pd.DataFrame,
    validation_size: float = 0.05,
    sample_size: Optional[int] = None,
    validation_by_slam: bool = True,
) -> pd.DataFrame:
    df = df.reset_index(drop=True)

    # Dates & match_id
    df["date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")
    df = df.sort_values(["date", "tourney_id", "match_num"])
    df["year"] = df["date"].dt.year
    df["rel_year"] = (df["date"].dt.year - df["date"].dt.year.min()) + 1
    df["time_since_epoch"] = df["date"].astype("int64") // 10**9
    # df = df[df["year"] > 2000]
    sets = df[["winner_name", "loser_name"]].apply(frozenset, axis=1)
    df["h2h_winner_player"] = df.groupby([sets, "winner_name"]).cumcount()
    df["h2h_loser_player"] = df.groupby(sets).cumcount() - df["h2h_winner_player"]
    cleaned_tourney = df["tourney_name"].str.split(" ").apply(lambda x: " ".join(x[:4]))
    biggest_tourney = cleaned_tourney.value_counts().sort_values(ascending=False)[:100]
    df["tournament"] = np.where(
        cleaned_tourney.isin(biggest_tourney.index), cleaned_tourney, "Other tournament"
    )
    df["match_id"] = df.index

    df["winner_age"] = df["winner_age"].fillna(25)
    df["loser_age"] = df["loser_age"].fillna(25)

    df["winner_rank"] = pd.cut(
        df["winner_rank"].fillna(0),
        bins=(-1, 0, 1, 2, 3, 4, 5, 10, 25, 50, float("inf")),
    ).astype(str)
    df["loser_rank"] = pd.cut(
        df["loser_rank"].fillna(0),
        bins=(-1, 0, 1, 2, 3, 4, 5, 10, 25, 50, float("inf")),
    ).astype(str)

    # 2 means left wins, 1 means right wins (doesn't appear in the dataset so we will randomly swap in __get_item__)
    df["target"] = 2
    if sample_size is not None:
        df = sample_df_by_match_id(df, sample_size)

    if validation_by_slam:
        # Define Grand Slam names
        grand_slams = ["australian open", "roland garros", "wimbledon", "us open"]
        is_slam = cleaned_tourney.str.lower().str.contains("|".join(grand_slams))
        tourney_id = df.groupby(is_slam)["tourney_id"].last().loc[True]

        # Mark validation rows
        df["is_validation"] = df["tourney_id"].eq(tourney_id)
        last_slam_index = df[df["tourney_id"] == tourney_id].index[-1]

        # Mark validation rows (this marks the last Grand Slam matches)
        df["is_validation"] = df["tourney_id"].eq(tourney_id)

        # Now filter out rows that come after the last index of the last Grand Slam
        df = df.loc[:last_slam_index].copy()
        # Drop rows after the last validation slam
        # df = df[df["date"] <= max_val_date].copy()

    else:
        val_size = int(len(df) * (1 - validation_size))
        validation_match_ids = df[val_size:]["match_id"].unique()
        df["is_validation"] = df["match_id"].isin(validation_match_ids)

    return df
