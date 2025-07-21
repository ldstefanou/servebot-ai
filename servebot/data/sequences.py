from collections import defaultdict
from heapq import merge
from typing import Dict, List

import numpy as np
import pandas as pd
import tqdm
from data.utils import merge_dicts_sorted, truncate_and_pad_to_long_tensor
from numpy.lib.stride_tricks import sliding_window_view


class PlayerIndex:
    def __init__(self, index: Dict[str, int]):
        self.index = index

    def get_match_idx_by_player(self, player: str):
        return self.index[player]

    def get_match_sequence_by_player(self, player: str):
        return np.arange(1, len(self.index[player]) + 1)

    def get_age_by_player(self, player: str):
        return np.arange(1, len(self.index[player]) + 1)


def create_player_index(df: pd.DataFrame) -> Dict[str, List[int]]:
    winner_matches = df.groupby("winner_name").indices
    loser_matches = df.groupby("loser_name").indices
    index = merge_dicts_sorted(winner_matches, loser_matches)
    return PlayerIndex(index)


def create_sliding_window_sequences(df: pd.DataFrame, keys: List[str], seq_length: int):
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
    containers = defaultdict(list)
    arrays = {}
    for col in keys:
        arrays[col] = df[col].values

    player_index = create_player_index(df)
    for player, rows in player_index.items():
        for key, arr in arrays.items():
            for window in sliding_window_view(rows, min(seq_length, len(rows))):
                containers[key].append(arr[window])

    all_tokens = {}
    for key, container in tqdm.tqdm(
        containers.items(), desc="Creating player centric sequence tensors"
    ):
        all_tokens[key] = truncate_and_pad_to_long_tensor(container, seq_length)

    return all_tokens


def create_match_specific_sequences(df: pd.DataFrame, keys: List[str], seq_length: int):
    print(f"Creating match sequences for {len(df)} matches...")
    containers = defaultdict(list)
    arrays = {}
    for col in keys:
        arrays[col] = df[col].values

    player_index: PlayerIndex = create_player_index(df)

    for idx, row in df.iterrows():
        player_1_matches = player_index.get_match_idx_by_player(row["winner_name"])
        player_2_matches = player_index.get_match_idx_by_player(row["loser_name"])

        # player_1_career = player_index.get_match_sequence_by_player(row["winner_name"])
        # player_2_career = player_index.get_match_sequence_by_player(row["loser_name"])

        merged_rows = np.asarray(list(merge(player_1_matches, player_2_matches)))
        unique_matches, unq_idx = np.unique(merged_rows, return_index=True)

        is_validation = arrays["is_validation"][unique_matches]
        is_match_in_past = unique_matches < idx
        is_historic_not_validation = np.logical_and(is_match_in_past, ~is_validation)
        unique_matches_before_this_match = unique_matches[is_match_in_past]

        # merged_rows =
        merged_rows = np.concatenate([unique_matches_before_this_match, [idx]])[
            -seq_length:
        ]

        for key, arr in arrays.items():
            containers[key].append(arr[merged_rows])
        containers["position_token"].append(np.arange(1, len(merged_rows) + 1))

    all_tokens = {}
    for key, container in tqdm.tqdm(
        containers.items(), desc="Creating match centric sequence tensors"
    ):
        all_tokens[key] = truncate_and_pad_to_long_tensor(container, seq_length)

    return all_tokens
