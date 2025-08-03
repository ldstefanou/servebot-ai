from typing import Dict

import torch
import torch.nn as nn
import yaml
from model.time_encoding import ContinuousTimeEncoding
from model.transformer import Transformer


def create_player_attention_mask(winner_tokens, loser_tokens, position_tokens):
    """Create attention mask allowing only player-specific context.

    Each match can only attend to previous matches involving the same players,
    preventing spurious correlations between unrelated matches.

    Parameters
    ----------
    winner_tokens : torch.Tensor
        Winner player tokens, shape [batch, seq_len]
    loser_tokens : torch.Tensor
        Loser player tokens, shape [batch, seq_len]

    position_tokens : torch.Tensor
        position tokens, shape [batch, seq_len]

    Returns
    -------
    torch.Tensor
        Boolean attention mask, shape [batch, seq_len, seq_len]
    """
    batch_size, seq_len = winner_tokens.shape

    # Create position-based causal mask: attend to earlier positions + diagonal
    pos_i = position_tokens.unsqueeze(2)  # [batch, seq_len, 1]
    pos_j = position_tokens.unsqueeze(1)  # [batch, 1, seq_len]

    # Allow attention to earlier positions (pos_j <= pos_i)
    causal_mask = pos_j <= pos_i

    # Expand tokens for broadcasting
    winner_i = winner_tokens.unsqueeze(2)  # [batch, seq_len, 1]
    loser_i = loser_tokens.unsqueeze(2)  # [batch, seq_len, 1]
    winner_j = winner_tokens.unsqueeze(1)  # [batch, 1, seq_len]
    loser_j = loser_tokens.unsqueeze(1)  # [batch, 1, seq_len]

    # Check all player combinations
    same_player_mask = (
        (winner_i == winner_j)  # same winner
        | (winner_i == loser_j)  # winner_i == loser_j
        | (loser_i == winner_j)  # loser_i == winner_j
        | (loser_i == loser_j)  # same loser
    )

    # Combine with causal mask
    not_padding = winner_tokens.ne(0)
    attention_mask = (
        same_player_mask
        & causal_mask
        & not_padding.unsqueeze(1)
        & not_padding.unsqueeze(2)
    )
    return attention_mask


class TennisTransformer(Transformer):
    def __init__(self, config, embeddings: Dict[str, Dict[str, int]]):
        super().__init__(config)
        self.embedding_dict = nn.ModuleDict()
        self.time = ContinuousTimeEncoding(config["d_model"])
        # Standard embeddings
        for key, mapping in embeddings.items():
            self.embedding_dict[key] = nn.Embedding(
                len(mapping), config["d_model"], padding_idx=0
            )

        self.embedding_dict["position"] = nn.Embedding(
            config["seq_length"] + 1, config["d_model"]
        )

        # Multi-dimensional projections for complex features
        self.h2h_projection = nn.Linear(2, config["d_model"])  # [h2h_diff, h2h_total]
        self.age_projection = nn.Linear(2, config["d_model"])  # [age_diff, avg_age]

        # Learnable scaling parameters
        self.h2h_scale = nn.Parameter(torch.tensor([1.0, 1.0]))
        self.age_scale = nn.Parameter(torch.tensor([1.0, 1.0]))

        self.to_pred = nn.Sequential(nn.Linear(config["d_model"], 2))

    def forward(self, batch):
        # Get basic embeddings
        embeddings = {}
        time = self.time(batch["time_since_epoch"])
        w_age = self.time(batch["winner_age"])
        l_age = self.time(batch["loser_age"])
        for key, embedder in self.embedding_dict.items():
            if key.startswith("player_"):
                feature = key.removeprefix("player_")
                embeddings[f"winner_{feature}"] = embedder(
                    batch[f"winner_{feature}_token"]
                )
                embeddings[f"loser_{feature}"] = embedder(
                    batch[f"loser_{feature}_token"]
                )
            else:
                embeddings[key] = embedder(batch[f"{key}_token"])

        # H2H encoding
        h2h_diff = (batch["h2h_winner_player"] - batch["h2h_loser_player"]).float()
        h2h_total = (batch["h2h_winner_player"] + batch["h2h_loser_player"]).float()
        h2h_features = torch.stack([h2h_diff, h2h_total], dim=-1)
        h2h_emb = self.h2h_projection(h2h_features * self.h2h_scale)

        # Age encoding - both difference and average matter
        age_diff = (batch["winner_age"] - batch["loser_age"]).float()
        avg_age = (batch["winner_age"] + batch["loser_age"]).float() / 2
        age_features = torch.stack([age_diff, avg_age], dim=-1)
        age_emb = self.age_projection(age_features * self.age_scale)

        # Player embeddings with advantages
        left_player_emb = embeddings.pop("winner_name") + w_age
        right_player_emb = embeddings.pop("loser_name") + l_age

        left_player_rank = embeddings.pop("winner_rank")
        right_player_rank = embeddings.pop("loser_rank")

        # Relative strength
        rel_strength = left_player_emb - right_player_emb
        rel_rank = left_player_rank - right_player_rank
        pos = embeddings.pop("position")

        rel_strength = left_player_emb - right_player_emb
        # Stack remaining features
        features = torch.stack(
            [
                rel_strength,
                rel_rank,
                age_emb,
                h2h_emb,
                embeddings["year"],
                embeddings["surface"],
                embeddings["round"],
                embeddings["tournament"],
            ],
            dim=-1,
        )
        x = features.mean(-1)
        attention_mask = create_player_attention_mask(
            batch[f"winner_name_token"],
            batch[f"loser_name_token"],
            batch["position_token"],
        )
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x + time, attention_mask)

        return self.to_pred(x)

    @classmethod
    def from_saved_artifacts(
        cls, model_path: str, embeddings: Dict[str, Dict[str, int]]
    ):
        """Load model from saved artifacts directory"""
        # Load config
        with open(f"{model_path}/model_config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Create model instance
        model = cls(config, embeddings)

        # Load weights
        weights = torch.load(f"{model_path}/model_weights.pt", map_location="cpu")
        model.load_state_dict(weights)

        return model
