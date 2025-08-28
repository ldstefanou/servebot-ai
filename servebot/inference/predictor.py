import pickle
from datetime import datetime

import torch
import yaml

from servebot.data.dataset import SimpleDSet
from servebot.data.encodings import apply_encoders
from servebot.data.preprocess import load_data
from servebot.data.sequences import create_match_specific_sequences
from servebot.data.utils import find_last_valid_positions
from servebot.inference.categories import get_categories
from servebot.model.index import PlayerIndex
from servebot.model.servebot import TennisTransformer
from servebot.paths import get_model_path


class MatchPredictor:
    def __init__(self, model, config, encoders, training_df):
        self.model = model
        self.config = config
        self.encoders = encoders
        self.training_df = training_df
        self.index = PlayerIndex(training_df)
        self.categories = get_categories(encoders)

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

        # Load training data
        training_df = load_data(filter_matches=config["dataset"], last=1000)
        training_df = apply_encoders(training_df, encoders)

        return cls(model, config, encoders, training_df)

    def predict(self, winner_name, loser_name, **kwargs):
        """Predict match outcome probability."""
        # Add today's date if not specified
        if "date" not in kwargs:
            kwargs["date"] = datetime.now()

        # Create synthetic match using existing PlayerIndex logic
        synthetic_match = self.index.create_match_for_players(
            winner_name, loser_name, self.encoders, **kwargs
        )
        synthetic_match_id = synthetic_match["match_id"].iloc[0]

        # Generate sequences with synthetic match included
        sequences = create_match_specific_sequences(
            self.index.df, self.index, self.config["columns"], self.config["seq_length"]
        )

        # Create dataset and find sequence containing our synthetic match
        dataset = SimpleDSet(sequences, training=False)

        # Find which sequence contains our synthetic match
        match_id_tensor = sequences["match_id"]
        synthetic_mask = match_id_tensor.eq(synthetic_match_id)
        sequence_idx = synthetic_mask.any(dim=1).nonzero(as_tuple=True)[0]

        if len(sequence_idx) == 0:
            raise ValueError(
                f"Synthetic match {synthetic_match_id} not found in sequences"
            )

        # Get the sequence and predict
        batch = dataset[sequence_idx[0].item()]
        batch = {k: v.unsqueeze(0) for k, v in batch.items()}

        with torch.no_grad():
            out = self.model(batch)
            probs = out.softmax(-1)

            # Find position of our synthetic match in the sequence
            is_padded = batch["winner_name_token"].eq(0)
            last_valid_pos = find_last_valid_positions(is_padded)[0].item()

            winner_prob = probs[0, last_valid_pos, 1].item()

        return winner_prob
