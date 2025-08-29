from typing import Dict, Optional

import torch
from torch.utils.data import Dataset

from servebot.data.utils import find_last_valid_positions


class SimpleDSet(Dataset):
    def __init__(self, sequences, training: bool = True):
        self.training = training
        self.sequences = sequences

    def __len__(self):
        return len(list(self.sequences.values())[0])

    def set_validation_mask(self, match_id):
        self.sequences["is_validation"] = self.sequences["match_id"].ge(match_id)

    def set_validation_mask_by_indexes(self, match_indexes):
        """Set validation mask based on specific match indexes from the original dataframe"""
        # Create a boolean mask for validation matches
        validation_match_mask = torch.zeros_like(
            self.sequences["match_id"], dtype=torch.bool
        )

        # Find sequence positions that contain any of the validation match indexes
        for match_idx in match_indexes:
            validation_match_mask |= self.sequences["match_id"].eq(match_idx)

        # Determine padding (non-padded positions)
        is_padded = self.sequences["winner_name_token"].eq(0)

        # Find the last non-padded position for each sequence
        last_valid_pos = find_last_valid_positions(is_padded)
        non_padded = ~is_padded

        # Check if the last non-padded position is a validation match
        batch_indices = torch.arange(validation_match_mask.size(0))
        last_pos_is_validation = validation_match_mask[batch_indices, last_valid_pos]

        # Only return sequences where the last valid match is a validation match
        valid_sequence_indices = last_pos_is_validation.nonzero(as_tuple=True)[
            0
        ].tolist()

        # Set validation mask for all positions after validation matches start
        self.sequences["is_validation"] = (
            validation_match_mask.cumsum(dim=1).gt(0) & non_padded
        )

        return valid_sequence_indices

    def __getitem__(self, idx):
        batch = {k: v[idx].clone() for k, v in self.sequences.items()}
        batch["target"] = torch.zeros_like(batch["winner_name_token"])
        batch["is_padding"] = batch["winner_name_token"].eq(0)
        if self.training:
            batch["loss_mask"] = torch.logical_and(
                ~batch["is_padding"], ~batch["is_validation"]
            )
            swap_batch_keys(batch, mask=~batch["is_padding"])
        return batch


def swap_batch_keys(
    batch: Dict,
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
    swap_pairs = [
        (k, k.replace("winner", "loser")) for k in batch.keys() if "winner" in k
    ]
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
