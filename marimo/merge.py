# /// script
# dependencies = ["pandas", "numpy"]
# ///

import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell
def _(pd):
    # Load both datasets for merge exploration

    print("=== LOADING DATASETS ===")

    # New dataset with betting odds (tennis-data.co.uk)
    df_new = pd.read_parquet("data/tennis_data.parquet")

    # ATP dataset with rich player metadata
    df_atp = pd.read_parquet("servebot/data/static/atp_matches_all_levels.parquet")
    print(f"ATP dataset: {df_atp.shape[0]} matches, {df_atp.shape[1]} columns")
    print(f"Odds dataset: {df_new.shape[0]} matches, {df_new.shape[1]} columns")
    return df_atp, df_new


@app.cell
def _(df_new):
    df_new.tail(10)
    return


@app.cell
def _(df_atp):
    df_atp
    return


@app.cell
def _(df_atp, df_new):
    # Explore name normalization possibilities

    print("=== NAME FORMAT ANALYSIS ===")

    # Sample names from each dataset
    new_winners = df_new["Winner"].dropna().head(20).tolist()
    atp_winners = df_atp["winner_name"].dropna().head(20).tolist()

    print("New dataset names (Last F.):")
    for name in new_winners[:10]:
        print(f"  {name}")

    print("\nATP dataset names (First Last):")
    for name in atp_winners[:10]:
        print(f"  {name}")

    # Try to find some common players
    print("\n=== COMMON PLAYER DETECTION ===")

    # Look for Federer, Nadal, Djokovic
    famous_players = ["Federer", "Nadal", "Djokovic", "Murray"]

    for player in famous_players:
        # Find in new dataset
        new_matches = df_new[
            (df_new["Winner"].str.contains(player, case=False, na=False))
            | (df_new["Loser"].str.contains(player, case=False, na=False))
        ]

        # Find in ATP dataset
        atp_matches = df_atp[
            (df_atp["winner_name"].str.contains(player, case=False, na=False))
            | (df_atp["loser_name"].str.contains(player, case=False, na=False))
        ]

        if len(new_matches) > 0 and len(atp_matches) > 0:
            # Get the actual name formats
            new_name = ""
            if df_new["Winner"].str.contains(player, case=False, na=False).any():
                new_name = new_matches[
                    df_new["Winner"].str.contains(player, case=False, na=False)
                ]["Winner"].iloc[0]
            else:
                new_name = new_matches[
                    df_new["Loser"].str.contains(player, case=False, na=False)
                ]["Loser"].iloc[0]

            atp_name = ""
            if df_atp["winner_name"].str.contains(player, case=False, na=False).any():
                atp_name = atp_matches[
                    df_atp["winner_name"].str.contains(player, case=False, na=False)
                ]["winner_name"].iloc[0]
            else:
                atp_name = atp_matches[
                    df_atp["loser_name"].str.contains(player, case=False, na=False)
                ]["loser_name"].iloc[0]

            print(f"{player}:")
            print(f"  New: '{new_name}' ({len(new_matches)} matches)")
            print(f"  ATP: '{atp_name}' ({len(atp_matches)} matches)")

    return


@app.cell
def _(df_atp, df_new, pd):
    # Analyze merge potential by date overlap

    print("=== DATE OVERLAP ANALYSIS ===")

    # Convert ATP dates to datetime
    df_atp_with_dates = df_atp.copy()
    df_atp_with_dates["date"] = pd.to_datetime(
        df_atp_with_dates["tourney_date"], format="%Y%m%d"
    )

    # Find date overlap
    new_min, new_max = df_new["Date"].min(), df_new["Date"].max()
    atp_min, atp_max = df_atp_with_dates["date"].min(), df_atp_with_dates["date"].max()

    overlap_start = max(new_min, atp_min)
    overlap_end = min(new_max, atp_max)

    print(f"New dataset: {new_min.date()} to {new_max.date()}")
    print(f"ATP dataset: {atp_min.date()} to {atp_max.date()}")
    print(f"Overlap period: {overlap_start.date()} to {overlap_end.date()}")

    # Filter both datasets to overlap period
    df_new_overlap = df_new[
        (df_new["Date"] >= overlap_start) & (df_new["Date"] <= overlap_end)
    ]
    df_atp_overlap = df_atp_with_dates[
        (df_atp_with_dates["date"] >= overlap_start)
        & (df_atp_with_dates["date"] <= overlap_end)
    ]

    print(f"\nMatches in overlap period:")
    print(f"  New dataset: {len(df_new_overlap)} matches")
    print(f"  ATP dataset: {len(df_atp_overlap)} matches")

    # Sample from overlap period
    print(f"\n=== OVERLAP SAMPLE (2023) ===")
    sample_2023_new = df_new_overlap[df_new_overlap["Date"].dt.year == 2023].head(5)
    sample_2023_atp = df_atp_overlap[df_atp_overlap["date"].dt.year == 2023].head(5)

    print("New dataset 2023 sample:")
    print(sample_2023_new[["Date", "Winner", "Loser", "Tournament"]].to_string())

    print("\nATP dataset 2023 sample:")
    print(
        sample_2023_atp[
            ["date", "winner_name", "loser_name", "tourney_name"]
        ].to_string()
    )

    return


@app.cell
def _(df_atp, df_new):
    # Show what metadata we'd gain from merging

    print("=== MERGE VALUE ANALYSIS ===")

    print("New dataset has (betting odds):")
    betting_cols = [
        col
        for col in df_new.columns
        if any(x in col for x in ["W", "L"])
        and any(y in col for y in ["CB", "GB", "B365", "Max", "Avg", "PS"])
    ]
    print(f"  {len(betting_cols)} betting columns: {betting_cols[:10]}...")

    print(f"\nATP dataset has (player metadata):")
    meta_cols = [
        col
        for col in df_atp.columns
        if any(x in col for x in ["age", "ht", "hand", "ioc", "rank"])
    ]
    print(f"  {len(meta_cols)} metadata columns: {meta_cols}")

    print(f"\nPotential combined dataset would have:")
    print(f"  - Recent matches with betting odds (New dataset strength)")
    print(f"  - Rich player metadata (ATP dataset strength)")
    print(f"  - Better date granularity (New dataset strength)")

    return


if __name__ == "__main__":
    app.run()
