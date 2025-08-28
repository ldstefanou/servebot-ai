import torch

from servebot.data.utils import find_last_valid_positions


def serialize_predictions_with_context(model, dataloader, df):
    """Serialize predictions with full match context from original dataframe."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            out = model(batch)
            proba = out.softmax(-1)
            pred = torch.argmax(out, dim=-1)

            # Extract tensor data
            match_ids = batch["match_id"].cpu().numpy()
            probabilities = proba[:, :, 1].cpu().numpy()
            predictions_binary = pred.cpu().numpy()
            targets = batch["target"].cpu().numpy()

            # Find last valid positions (non-padded)
            is_padded = batch["winner_name_token"].eq(0)
            last_valid_pos = find_last_valid_positions(is_padded)

            # Extract data only from last valid positions
            batch_size = match_ids.shape[0]
            for i in range(batch_size):
                j = last_valid_pos[i].item()  # Last valid position for this sequence
                predictions.append(
                    {
                        "match_id": match_ids[i, j],
                        "winner_prob": probabilities[i, j],
                        "loser_prob": 1 - probabilities[i, j],
                        "prediction": predictions_binary[i, j],
                        "target": targets[i, j],
                    }
                )

    df_pred = pd.DataFrame(predictions)

    # Merge with original dataframe for full context
    match_cols = [
        "match_id",
        "winner_name",
        "loser_name",
        "surface",
        "tournament",
        "round",
        "date",
        "winner_rank",
        "loser_rank",
    ]
    if "winner_odds_avg" in df.columns:
        match_cols.extend(["winner_odds_avg", "loser_odds_avg"])

    df_matches = df[match_cols].drop_duplicates(subset=["match_id"])
    df_pred = df_pred.merge(df_matches, on="match_id", how="left")

    df_pred.to_parquet("predictions.parquet")
    return df_pred
