# /// script
# dependencies = ["pandas", "numpy", "matplotlib"]
# ///

import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return np, pd, plt


@app.cell
def _(pd):
    org = pd.read_parquet("servebot/data/static/atp_matches_with_odds.parquet")
    org.tail(100)
    return


@app.cell
def _(pd):
    df = pd.read_parquet("predictions.parquet")
    return (df,)


@app.cell
def _(df):
    is_wimby = df["tournament"] == "Wimbledon"
    is_last = df["date"].dt.year == 2025
    print(f"Predictions: {len(df)}")
    print(
        f"Accuracy: {(df[is_wimby & is_last]['prediction'] == df[is_wimby & is_last]['target']).mean():.3f}"
    )
    df.tail()
    return


@app.cell
def _(df):
    df.head(100)
    return


@app.cell
def _(df, np, plt):
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.hist(df["winner_prob"], bins=30)
    plt.title("Probabilities")

    plt.subplot(132)
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_indices = np.digitize(df["winner_prob"], bins) - 1

    actual = [
        (
            df.loc[bin_indices == i, "target"].mean()
            if (bin_indices == i).sum() > 0
            else np.nan
        )
        for i in range(len(bin_centers))
    ]

    plt.plot(bin_centers, actual, "o-")
    plt.plot([0, 1], [0, 1], "--", alpha=0.5)
    plt.title("Calibration")

    plt.subplot(133)
    acc_by_surf = df.groupby("surface").apply(
        lambda x: (x["prediction"] == x["target"]).mean()
    )
    acc_by_surf.plot(kind="bar")
    plt.title("Accuracy by Surface")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df):
    if "winner_odds_avg" in df.columns:
        odds = df.dropna(subset=["winner_odds_avg", "loser_odds_avg"])
        odds["implied"] = 1 / odds["winner_odds_avg"]
        odds["total"] = odds["implied"] + 1 / odds["loser_odds_avg"]
        odds["implied_norm"] = odds["implied"] / odds["total"]

        corr = odds["winner_prob"].corr(odds["implied_norm"])
        print(f"Model vs Odds correlation: {corr:.3f}")

        odds[["winner_prob", "implied_norm", "target"]].head()
    return (odds,)


@app.cell
def _(df):
    wrong = df[
        (df["prediction"] != df["target"]) & (abs(df["winner_prob"] - 0.5) > 0.3)
    ]
    print(f"Confident mistakes: {len(wrong)}")
    if len(wrong) > 0:
        wrong[["winner_name", "loser_name", "winner_prob", "surface"]].head()
    return


@app.cell
def _(odds):
    odds["winner_prob"]
    return


@app.cell
def _(odds):
    odds
    return


if __name__ == "__main__":
    app.run()
