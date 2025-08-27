import argparse

import torch
import yaml
from data.dataset import SimpleDSet
from data.embeddings import apply_encoders, create_encoders
from data.preprocess import get_grand_slam_match_indexes, load_data
from data.sequences import create_match_specific_sequences
from data.utils import set_seed
from model.index import PlayerIndex
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from train import train

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

    args = parser.parse_args()

    if args.last and args.sample:
        raise ValueError("Cannot set both sample and last, pick one")

    with open("model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # Load model configuration

    df = load_data(from_year=args.from_year, filter_matches="atp", last=args.last)
    encoders = create_encoders(df, config["embedders"])
    df = apply_encoders(df, encoders)

    index = PlayerIndex(df)
    sequences = create_match_specific_sequences(
        df, index, config["columns"], config["seq_length"]
    )
    dset = SimpleDSet(sequences)

    validation = []
    for slam_index in (4, 3, 2, 1, 0):

        # Get Grand Slam match indexes for validation
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
        vdl = DataLoader(
            Subset(dset, val_indices),
            batch_size=64,
        )
        val_score = train(tdl, vdl, config, encoders)
        validation.append(val_score)
    print(validation)
