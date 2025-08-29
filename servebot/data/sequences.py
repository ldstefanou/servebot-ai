from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import tqdm
from numpy.lib.stride_tricks import sliding_window_view

from servebot.data.utils import truncate_and_pad_to_long_tensor
from servebot.model.index import PlayerIndex


def create_sliding_window_sequences(df: pd.DataFrame, keys: List[str], seq_length: int):
    """Create sequences using chronological sliding window."""
    containers = defaultdict(list)
    arrays = {}
    for col in keys:
        arrays[col] = df[col].values

    indexes = np.arange(len(df))

    for rows in tqdm.tqdm(
        sliding_window_view(indexes, seq_length), desc="Creating sequences.."
    ):
        for key, arr in arrays.items():
            containers[key].append(arr[np.sort(rows)])

    all_tokens = {}
    for key, container in tqdm.tqdm(
        containers.items(), desc="Creating sequence tensors"
    ):
        all_tokens[key] = truncate_and_pad_to_long_tensor(container, seq_length)

    return all_tokens


def create_player_specific_sequences(
    df: pd.DataFrame, keys: List[str], seq_length: int
):
    """Create sequences from each player's match history."""
    containers = defaultdict(list)
    arrays = {}
    for col in keys:
        arrays[col] = df[col].values

    player_index = PlayerIndex(df)
    for player, rows in player_index.index.items():
        for key, arr in arrays.items():
            for window in sliding_window_view(rows, min(seq_length, len(rows))):
                containers[key].append(arr[window])

    all_tokens = {}
    for key, container in tqdm.tqdm(
        containers.items(), desc="Creating player centric sequence tensors"
    ):
        all_tokens[key] = truncate_and_pad_to_long_tensor(container, seq_length)

    return all_tokens


def create_match_specific_sequences(
    player_index: PlayerIndex, keys: List[str], seq_length: int
):
    """Create sequences combining both players' histories for each match."""
    print(f"Creating match sequences for {len(player_index.df)} matches...")
    containers = defaultdict(list)

    for idx, row in tqdm.tqdm(
        player_index.df.iterrows(),
        total=len(player_index.df),
        desc="Creating match sequences",
    ):
        match_sequence = player_index.get_matches_by_players(
            row["winner_name"], row["loser_name"]
        )

        is_match_in_past = match_sequence < idx
        unique_matches_before_this_match = match_sequence[is_match_in_past]

        merged_rows = np.concatenate([unique_matches_before_this_match, [idx]])[
            -seq_length:
        ]

        for key in keys:
            containers[key].append(player_index.get_values_for_idx(key, merged_rows))
        containers["position_token"].append(np.arange(1, len(merged_rows) + 1))

    all_tokens = {}
    for key, container in tqdm.tqdm(
        containers.items(), desc="Creating match centric sequence tensors"
    ):
        all_tokens[key] = truncate_and_pad_to_long_tensor(container, seq_length)
    return all_tokens
