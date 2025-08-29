from typing import Optional

import numpy as np
import pandas as pd

from servebot.paths import get_dataset_path


def load_training_dataframe():
    df = pd.read_parquet(get_dataset_path("all"))
    df["date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")
    df = df.sort_values(["date", "tourney_id", "match_num"])
    return df[df["source"] == "atp"]


def load_odds_dataframe():
    """Load and transform tennis-data.co.uk odds data to match ATP format."""
    df = pd.read_parquet(get_dataset_path("odds"))

    # Create a mapping from odds data to ATP format
    df_transformed = pd.DataFrame()

    # Basic match info
    df_transformed["tourney_name"] = df["Tournament"]
    df_transformed["surface"] = df["Surface"]
    df_transformed["tourney_date"] = (
        pd.to_datetime(df["Date"]).dt.strftime("%Y%m%d").astype(int)
    )
    df_transformed["round"] = df["Round"]

    # Player info
    df_transformed["winner_name"] = df["Winner"]
    df_transformed["loser_name"] = df["Loser"]
    df_transformed["winner_rank"] = pd.to_numeric(df["WRank"], errors="coerce")
    df_transformed["loser_rank"] = pd.to_numeric(df["LRank"], errors="coerce")

    # Set info from individual sets
    df_transformed["score"] = ""
    for i in range(1, 6):  # Up to 5 sets
        w_col, l_col = f"W{i}", f"L{i}"
        if w_col in df.columns and l_col in df.columns:
            set_scores = df[w_col].astype(str) + "-" + df[l_col].astype(str)
            # Only add if both scores are not NaN
            mask = ~(pd.isna(df[w_col]) | pd.isna(df[l_col]))
            df_transformed.loc[mask, "score"] += set_scores[mask] + " "

    df_transformed["score"] = df_transformed["score"].str.strip()

    # Add betting odds (average of available bookmakers)
    winner_odds_cols = [
        "CBW",
        "GBW",
        "IWW",
        "SBW",
        "B365W",
        "B&WW",
        "EXW",
        "PSW",
        "UBW",
        "LBW",
        "SJW",
    ]
    loser_odds_cols = [
        "CBL",
        "GBL",
        "IWL",
        "SBL",
        "B365L",
        "B&WL",
        "EXL",
        "PSL",
        "UBL",
        "LBL",
        "SJL",
    ]

    # Calculate average odds, ignoring NaN values
    winner_odds = df[winner_odds_cols].apply(pd.to_numeric, errors="coerce")
    loser_odds = df[loser_odds_cols].apply(pd.to_numeric, errors="coerce")

    df_transformed["winner_odds"] = winner_odds.mean(axis=1, skipna=True)
    df_transformed["loser_odds"] = loser_odds.mean(axis=1, skipna=True)

    # Use max and avg odds if available
    if "MaxW" in df.columns and "MaxL" in df.columns:
        df_transformed["winner_odds_max"] = pd.to_numeric(df["MaxW"], errors="coerce")
        df_transformed["loser_odds_max"] = pd.to_numeric(df["MaxL"], errors="coerce")

    if "AvgW" in df.columns and "AvgL" in df.columns:
        df_transformed["winner_odds_avg"] = pd.to_numeric(df["AvgW"], errors="coerce")
        df_transformed["loser_odds_avg"] = pd.to_numeric(df["AvgL"], errors="coerce")

    # Add missing columns to match ATP format with defaults
    df_transformed["tourney_id"] = (
        df["year"].astype(str) + "-" + df["ATP"].astype(str).str.zfill(4)
    )
    df_transformed["draw_size"] = np.nan
    df_transformed["tourney_level"] = "ATP250"  # Default assumption
    df_transformed["match_num"] = range(1, len(df_transformed) + 1)

    # Player IDs and details (unknown for odds data)
    for prefix in ["winner", "loser"]:
        df_transformed[f"{prefix}_id"] = np.nan
        df_transformed[f"{prefix}_seed"] = np.nan
        df_transformed[f"{prefix}_entry"] = ""
        df_transformed[f"{prefix}_hand"] = "U"  # Unknown
        df_transformed[f"{prefix}_ht"] = np.nan
        df_transformed[f"{prefix}_ioc"] = ""
        df_transformed[f"{prefix}_age"] = 20
        df_transformed[f"{prefix}_rank_points"] = np.nan

    # Add source identifier
    df_transformed["source"] = "odds"
    df_transformed["year"] = df["year"]

    # Create date column for sorting
    df_transformed["date"] = pd.to_datetime(
        df_transformed["tourney_date"], format="%Y%m%d"
    )
    df_transformed = df_transformed.sort_values(["date", "tourney_id", "match_num"])

    return df_transformed


def sample_df_by_match_id(df: pd.DataFrame, size: int):
    unique_matches = df["match_id"].unique()
    sampled_matches = np.random.choice(
        unique_matches, size=min(size, len(unique_matches)), replace=False
    )
    df = df[df["match_id"].isin(sampled_matches)]
    return df


def bin_continuous_features(df: pd.DataFrame):
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


def preprocess_dataframe(df: pd.DataFrame):
    # time stuff
    df["year"] = df["date"].dt.year
    df["time_since_epoch"] = df["date"].astype("int64") // 10**9

    # h2h scores
    sets = df[["winner_name", "loser_name"]].apply(lambda x: frozenset(x), axis=1)
    df["h2h_winner_player"] = df.groupby([sets, "winner_name"]).cumcount()
    df["h2h_loser_player"] = df.groupby(sets).cumcount() - df["h2h_winner_player"]

    # match id
    df["match_id"] = np.arange(1, df.shape[0] + 1)

    # Apply categorical binning
    df = bin_continuous_features(df)

    df["tourney_name"] = df["tourney_name"].str.replace("Us Open", "US Open")
    # clean tournament name
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


def get_last_tournament_match_indexes(
    df: pd.DataFrame, tournament_index: int = 0, grand_slam_only: bool = True
):
    """
    Get dataframe indexes for the last tournament.

    Args:
        df: The tennis dataframe (assumed to be sorted by date)
        tournament_index: Index of tournament (0 = last/most recent, 1 = second to last, etc.)
        grand_slam_only: Whether to filter only Grand Slam tournaments

    Returns:
        List of dataframe indexes that belong to the selected tournament
    """
    if grand_slam_only:
        # Define Grand Slam names
        grand_slams = ["australian open", "roland garros", "wimbledon", "us open"]
        # Filter by Grand Slams
        tournament_filter = (
            df["tourney_str"].str.lower().str.contains("|".join(grand_slams))
        )
        filtered_df = df[tournament_filter].copy()
    else:
        # Use all tournaments
        filtered_df = df.copy()

    # Get the last tournament's info
    last_row = filtered_df.iloc[-(tournament_index + 1)]
    last_tourney = last_row["tourney_str"]
    last_year = last_row["year"]

    # Return all matches from that tournament
    mask = (filtered_df["tourney_str"] == last_tourney) & (
        filtered_df["year"] == last_year
    )
    return filtered_df[mask]["match_id"].tolist()


def load_data(
    sample: Optional[int] = None,
    last: Optional[int] = None,
    from_year: Optional[int] = None,
    filter_matches: str = "all",
):
    if filter_matches == "odds":
        df = load_odds_dataframe()
    else:
        df = load_training_dataframe()
    df = preprocess_dataframe(df)
    # Filter by match type
    if filter_matches == "atp":
        df = df[df["source"] == "atp"]

    if sample is not None:
        df = sample_df_by_match_id(df, sample)
    if last is not None:
        df = df.iloc[-last:].copy()
    if from_year is not None:
        df = df[df["year"].ge(from_year)].copy()
    return df.reset_index()
