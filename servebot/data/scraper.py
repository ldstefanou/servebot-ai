"""Simple tennis data scraper for tennis-data.co.uk historical data."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

import pandas as pd


def download_year(year: int) -> Optional[pd.DataFrame]:
    """Download tennis data for a single year."""
    url = f"http://tennis-data.co.uk/{year}/{year}.xls"
    try:
        df = pd.read_excel(url)
        df["year"] = year
        print(f"Downloaded {year}: {len(df)} matches")
        return df
    except Exception as e:
        print(f"Failed to download {year}: {e}")
        return None


def download_all_tennis_data() -> pd.DataFrame:
    """Download all tennis data from 2000 to current year using multithreading."""
    current_year = datetime.now().year
    years = range(2000, current_year + 1)

    # Download in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(download_year, years))

    # Filter out None results and combine
    valid_dfs = [df for df in results if df is not None]
    combined = pd.concat(valid_dfs, ignore_index=True)

    # Clean all numeric columns - convert problematic values to NaN
    for col in combined.columns:
        if col not in [
            "Date",
            "Tournament",
            "Series",
            "Court",
            "Surface",
            "Round",
            "Winner",
            "Loser",
            "Comment",
        ]:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Sort by date
    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined = combined.sort_values("Date")

    return combined


def main():
    df = download_all_tennis_data()
    df.to_parquet("data/tennis_data.parquet")
    print(f"Saved {len(df)} total matches")


if __name__ == "__main__":
    main()
