from typing import Dict, List, Optional

import pandas as pd
import torch
from data.embeddings import apply_embeddings
from data.sequences import *
from torch.utils.data import Dataset, Sampler


class TennisDataset(Dataset):
    def __init__(
        self,
        embeddings,
        df: pd.DataFrame,
        config,
        training_mode: bool = True,
    ):

        self.seq_length = config["seq_length"]
        self.columns = config["columns"]

        # Load data to create shared vocabulary

        self.embeddings = embeddings
        self.training_mode = training_mode
        self.sequence_type = config["sequence_type"]
        self.config = config  # Store config for saving
        self._create_sequences(df)

    def _create_sequences(self, df: pd.DataFrame):
        df = apply_embeddings(df, self.embeddings)
        index = PlayerIndex(df)
        # Use ALL data for sequences (train + validation)

        if self.sequence_type == "temporal":
            sequences = create_sliding_window_sequences(
                df, self.columns, self.seq_length
            )
        elif self.sequence_type == "match":
            sequences = create_match_specific_sequences(
                df, index, self.columns, self.seq_length
            )
        else:  # default to player
            sequences = create_player_specific_sequences(
                df, self.columns, self.seq_length
            )

        for key, seq in sequences.items():
            setattr(self, key, seq)
            self.size = len(seq)

    def __len__(self) -> int:
        return self.size

    @property
    def target_size(self):
        return getattr(self, "target").max() + 1

    def __getitem__(self, idx: int) -> Dict:
        # loss mask should be one where the loss should be backpropagated.
        # In our case that is the last match ;expect when its the validation match
        batch = {
            key: getattr(self, key)[idx].clone() for key in self.columns
        }  # Clone to avoid in-place modification
        batch["is_validation"] = batch["is_validation"].bool()
        batch["position_token"] = getattr(self, "position_token")[idx]

        # Add previous winner player context (before swapping)
        batch["previous_winner_player"] = torch.cat(
            [
                torch.zeros(1, dtype=batch["winner_name_token"].dtype),
                batch["winner_name_token"][:-1],
            ]
        )

        # Define what keys to swap when flipping left/right players
        swap_pairs = [
            ("winner_name_token", "loser_name_token"),
            ("h2h_winner_player", "h2h_loser_player"),
            ("winner_hand_token", "loser_hand_token"),
            ("winner_age", "loser_age"),
            ("winner_rank_token", "loser_rank_token"),
        ]
        player_1 = batch["winner_name_token"]
        batch["is_padding"] = player_1.eq(0)

        # Create target tensor (0 = left player wins, 1 = padding)
        batch["target"] = torch.zeros_like(batch["winner_name_token"])

        # Only swap non-padding positions
        non_padding_mask = ~batch["is_padding"]
        if self.training_mode:
            swap_batch_keys(batch, swap_pairs, mask=non_padding_mask)
        loss_mask = torch.logical_and(~batch["is_validation"], ~batch["is_padding"])
        batch["loss_mask"] = loss_mask
        return batch

    def save_model_artifacts(self, model, save_dir="model_artifacts"):
        import pickle
        from pathlib import Path

        import torch
        import yaml

        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        torch.save(model.state_dict(), save_path / "model_weights.pt")

        # Save config dict as yaml
        with open(save_path / "model_config.yaml", "w") as f:
            yaml.dump(self.config, f)

        with open(save_path / "embeddings.pkl", "wb") as f:
            pickle.dump(self.embeddings, f)

        return save_path


class ValSampler(Sampler):
    def __init__(self, dataset):
        # Get all is_validation tensors and find sequences with ANY validation examples
        any_is_val = dataset.is_validation.any(dim=1)
        self.val_indices = any_is_val.nonzero(as_tuple=True)[0].tolist()

    def __iter__(self):
        return iter(self.val_indices)

    def __len__(self):
        return len(self.val_indices)


def swap_batch_keys(
    batch: Dict,
    swap_pairs: List[tuple],
    prob: float = 0.5,
    mask: Optional[torch.Tensor] = None,
):
    """
    Randomly swap specified key pairs in batch with given probability per sequence position.

    Args:
        batch: Dictionary containing tensors
        swap_pairs: List of (key1, key2) tuples to swap
        prob: Probability of swapping each position
        mask: Boolean mask for positions to consider for swapping
    """
    if not swap_pairs:
        return

    # Get sequence length from first key
    first_key = next(iter(batch.keys()))
    seq_len = batch[first_key].shape[0]

    # Create swap mask, but only for non-masked positions
    if mask is not None:
        swap_mask = torch.rand(seq_len) < prob
        swap_mask = swap_mask & mask  # Only swap where mask is True
    else:
        swap_mask = torch.rand(seq_len) < prob

    # Swap each pair using the same mask
    for key1, key2 in swap_pairs:
        batch[key1][swap_mask], batch[key2][swap_mask] = (
            batch[key2][swap_mask],
            batch[key1][swap_mask],
        )

    # Flip targets: 0->1 for swapped positions (non-swapped stay as 0)
    batch["target"][swap_mask] = torch.where(
        batch["target"][swap_mask] == 0, 1, batch["target"][swap_mask]
    )
