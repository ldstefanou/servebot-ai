import pickle
from typing import List

import torch
import yaml

from servebot.data.dataset import SimpleDSet
from servebot.data.sequences import create_match_specific_sequences
from servebot.data.utils import find_last_valid_positions
from servebot.metrics import print_sequence_results
from servebot.model.index import PlayerIndex
from servebot.model.servebot import TennisTransformer
from servebot.models import Match
from servebot.paths import get_model_path


class MatchPredictor:
    def __init__(self, model, config, encoders):
        self.model = model
        self.config = config
        self.encoders = encoders

    @classmethod
    def load(cls, model_name="epoch_4"):
        """Load model and training data from checkpoint."""
        model_path = get_model_path(model_name)

        # Load config
        with open(model_path / "model_config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Load encoders
        with open(model_path / "embeddings.pkl", "rb") as f:
            encoders = pickle.load(f)

        # Load model
        model = TennisTransformer(config, encoders)
        model.load_state_dict(torch.load(model_path / "model_weights.pt"))
        model.eval()

        return cls(model, config, encoders)

    def predict(self, matches: List[Match]):
        """Predict match outcome probability."""

        # Create synthetic match using existing PlayerIndex logic
        synthetic_match = Match.to_df(matches, self.encoders)

        index = PlayerIndex(synthetic_match)
        # Generate sequences with synthetic match included
        sequences = create_match_specific_sequences(
            index, self.config["columns"], self.config["seq_length"]
        )

        # Create dataset and find sequence containing our synthetic match
        dataset = SimpleDSet(sequences, training=False)

        # Get the sequence and predict
        batch = dataset[-1]
        batch = {k: v.unsqueeze(0) for k, v in batch.items()}

        with torch.no_grad():
            out = self.model(batch)
            probs = out.softmax(-1)

            # Find position of our synthetic match in the sequence
            is_padded = batch["winner_name_token"].eq(0)
            last_valid_pos = find_last_valid_positions(is_padded)[0].item()

            winner_prob = probs[0, last_valid_pos, 1].item()
        print_sequence_results(batch, out, self.encoders, print_last=False)
        return winner_prob
