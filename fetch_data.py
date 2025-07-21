#!/usr/bin/env python3

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import StringIO

import pandas as pd
import requests


def get_atp_file_list():
    """Get list of ATP CSV files from GitHub API"""
    api_url = "https://api.github.com/repos/JeffSackmann/tennis_atp/contents"
    response = requests.get(api_url)
    response.raise_for_status()

    files = []
    for item in response.json():
        if item["type"] == "file" and item["name"].endswith(".csv"):
            name = item["name"]
            # Include main tour, futures, and qualifiers (exclude doubles)
            if (
                name.startswith("atp_matches_")
                and not "doubles" in name
                and (
                    "futures" in name
                    or "qual_chall" in name
                    or any(
                        name.startswith(f"atp_matches_{year}")
                        for year in range(1968, 2025)
                    )
                )
            ):
                files.append(name)

    return sorted(files)


def download_file(filename, base_url):
    """Download a single CSV file and return DataFrame"""
    try:
        url = base_url + filename
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Read CSV from string
        df = pd.read_csv(StringIO(response.text))

        # Add source column based on filename
        if "futures" in filename:
            df["source"] = "futures"
        elif "qual_chall" in filename:
            df["source"] = "quali"
        else:
            df["source"] = "atp"

        print(f"✅ Downloaded {filename} ({len(df)} {df['source'].iloc[0]} matches)")
        return df

    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        return None


def download_and_combine_atp_data(max_workers=8):
    """Download ATP CSV files in parallel and combine them"""

    base_url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/"

    print("Getting list of ATP files...")
    try:
        csv_files = get_atp_file_list()
    except Exception as e:
        print(f"Error getting file list: {e}")
        return None

    print(f"Found {len(csv_files)} ATP CSV files")
    print(f"Downloading with {max_workers} parallel workers...")

    # Download files in parallel
    download_func = partial(download_file, base_url=base_url)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        dataframes = list(executor.map(download_func, csv_files))

    # Filter out None values (failed downloads)
    dataframes = [df for df in dataframes if df is not None]

    if dataframes:
        print("Combining all dataframes...")
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Convert problematic columns to string to avoid type conflicts
        combined_df["tourney_level"] = combined_df["tourney_level"].astype(str)

        # Save to parquet
        output_file = "atp_matches_all_levels.parquet"
        combined_df.to_parquet(output_file, index=False)

        print(f"Combined {len(combined_df)} matches into {output_file}")
        print(
            f"Date range: {combined_df['tourney_date'].min()} to {combined_df['tourney_date'].max()}"
        )
        print(
            f"Years covered: {len(combined_df['tourney_date'].astype(str).str[:4].unique())} years"
        )

        return combined_df
    else:
        print("No data to combine")
        return None


if __name__ == "__main__":
    df = download_and_combine_atp_data()
    if df is not None:
        print("\nData successfully downloaded and processed!")
    else:
        print("Failed to fetch data.")
