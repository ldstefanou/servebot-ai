from collections import defaultdict
from functools import cached_property
from heapq import merge
from typing import List

import numpy as np
import pandas as pd
import tqdm
from data.utils import merge_dicts_sorted, truncate_and_pad_to_long_tensor
from numpy.lib.stride_tricks import sliding_window_view


class PlayerIndex:
    def __init__(self, df: pd.DataFrame):
        df = df.reset_index(drop=True)

        winner_matches = df.groupby("winner_name").indices
        loser_matches = df.groupby("loser_name").indices
        self.index = merge_dicts_sorted(winner_matches, loser_matches)
        self.df = df

    def get_last_value_for_player(self, player: str, key: str):
        vals = self.index[player]
        return self.df[key].iloc[vals[-1]]

    @cached_property
    def array_containers(self):
        return {k: self.df[k].values for k in self.df.columns}

    def get_values_for_idx(self, key, idx):
        return self.array_containers[key][idx]

    @property
    def players(self):
        return list(self.index.keys())

    def get_match_idx_by_player(self, player: str):
        return self.index[player]

    def get_matches_by_players(self, player_1: str, player_2: str):
        left = self.get_match_idx_by_player(player_1)
        right = self.get_match_idx_by_player(player_2)
        merged = np.unique(list(merge(left, right)))
        return merged


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
    match_df: pd.DataFrame, player_index: PlayerIndex, keys: List[str], seq_length: int
):
    """Create sequences combining both players' histories for each match."""
    print(f"Creating match sequences for {len(match_df)} matches...")
    containers = defaultdict(list)

    for idx, row in tqdm.tqdm(
        match_df.iterrows(), total=len(match_df), desc="Creating match sequences"
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
