import argparse

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

from servebot.data.dataset import SimpleDSet
from servebot.data.embeddings import apply_encoders, create_encoders
from servebot.data.preprocess import get_grand_slam_match_indexes, load_data
from servebot.data.sequences import create_match_specific_sequences
from servebot.data.utils import find_last_valid_positions, set_seed
from servebot.model.index import PlayerIndex
from servebot.model.servebot import TennisTransformer
from servebot.paths import get_model_path
from servebot.train import train


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


if __name__ == "__main__":
    set_seed()
    parser = argparse.ArgumentParser(description="Tennis prediction training")
    parser.add_argument(
        "--sample", type=int, default=None, help="Use only N samples for dry run"
    )
    parser.add_argument(
        "--last", type=int, default=None, help="Use only last N samples for dryrun"
    )
    parser.add_argument(
        "--from-year", type=int, default=None, help="Use sample from year x onwards"
    )
    parser.add_argument(
        "--validation-leak-test",
        action="store_true",
        help="Flip match outcomes for validation leak test",
    )
    parser.add_argument(
        "--use-validation",
        action="store_true",
        default=True,
        help="Use validation splits (default: True). Set --no-use-validation to train on full dataset",
    )
    parser.add_argument(
        "--no-use-validation",
        dest="use_validation",
        action="store_false",
        help="Train on full dataset without validation",
    )

    args = parser.parse_args()

    if args.last and args.sample:
        raise ValueError("Cannot set both sample and last, pick one")

    with open("model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # Load model configuration

    df = load_data(
        from_year=args.from_year, filter_matches=config["dataset"], last=args.last
    )
    encoders = create_encoders(df, config["embedders"])
    df = apply_encoders(df, encoders)

    index = PlayerIndex(df)
    sequences = create_match_specific_sequences(
        df, index, config["columns"], config["seq_length"]
    )
    dset = SimpleDSet(sequences)

    if args.use_validation:
        validation = []
        vdl = None

        for slam_index in (0,):
            val_match_indexes = get_grand_slam_match_indexes(df, slam_index=slam_index)
            print(
                f"Using Grand Slam index {slam_index} with {len(val_match_indexes)} matches for validation"
            )

            val_indices = dset.set_validation_mask_by_indexes(val_match_indexes)
            train_indices = torch.arange(0, max(val_indices)).tolist()

            tdl = DataLoader(
                dset,
                batch_size=config["batch_size"],
                sampler=SubsetRandomSampler(train_indices),
            )
            vdl = DataLoader(Subset(dset, val_indices), batch_size=64)

            val_score = train(
                tdl,
                vdl,
                config,
                encoders,
                validation_leak_test=args.validation_leak_test,
            )
            validation.append(val_score)

        print(validation)

        # Serialize predictions
        print("Serializing predictions...")
        model_path = get_model_path(f"epoch_{config['epochs']-1}")
        model = TennisTransformer(config, encoders)
        model.load_state_dict(torch.load(model_path / "model_weights.pt"))

    else:
        print("Training on full dataset without validation")
        tdl = DataLoader(dset, batch_size=config["batch_size"], shuffle=True)
        vdl = DataLoader(dset, batch_size=64)  # Use same data for validation

        train(
            tdl, vdl, config, encoders, validation_leak_test=args.validation_leak_test
        )
        print("Training completed on full dataset")

    dset.training = False
    dl = DataLoader(dset, batch_size=config["batch_size"])
    df_pred = serialize_predictions_with_context(model, dl, df)
    print(f"Saved predictions: {len(df_pred)} predictions")
