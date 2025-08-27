from typing import Dict, Optional

import torch
from torch.utils.data import Dataset, Sampler


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
        non_padded = self.sequences["winner_name_token"].ne(0)

        # Find the last non-padded position for each sequence
        last_valid_pos = non_padded.flip(dims=[1]).long().argmax(dim=1)
        last_valid_pos = non_padded.size(1) - 1 - last_valid_pos

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
        if self.training:
            batch["is_padding"] = batch["winner_name_token"].eq(0)
            batch["loss_mask"] = torch.logical_and(
                ~batch["is_padding"], ~batch["is_validation"]
            )
            batch["target"] = torch.zeros_like(batch["winner_name_token"])
            swap_batch_keys(batch, mask=~batch["is_padding"])
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
