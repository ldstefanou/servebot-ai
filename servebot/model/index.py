from functools import cached_property
from heapq import merge

import numpy as np
import pandas as pd

from servebot.data.embeddings import apply_encoders
from servebot.data.utils import merge_dicts_sorted


class PlayerIndex:
    def __init__(self, df: pd.DataFrame):
        self.create_index(df)

    def create_index(self, df):
        df = df.reset_index(drop=True)

        winner_matches = df.groupby("winner_name").indices
        loser_matches = df.groupby("loser_name").indices

        self.index = merge_dicts_sorted(winner_matches, loser_matches)
        self.df = df

    def get_last_value_for_player(self, player: str, key: str):
        idx = self.index[player][-1]
        row = self.df.iloc[idx]
        if "winner" in key:
            if row["winner_name"] == player:
                return row[key]
            else:
                return row[key.replace("winner", "loser")]
        elif "loser" in key:
            if row["loser_name"] == player:
                return row[key]
            else:
                return row[key.replace("loser", "winner")]
        return row[key]

    @cached_property
    def array_containers(self):
        return {k: self.df[k].values for k in self.df.columns}

    def get_values_for_idx(self, key, idx):
        return self.array_containers[key][idx]

    @property
    def players(self):
        return list(self.index.keys())

    def get_match_ids_by_player(self, player: str):
        return self.index[player]

    def get_matches_by_players(self, player_1: str, player_2: str):
        left = self.get_match_ids_by_player(player_1)
        right = self.get_match_ids_by_player(player_2)
        merged = np.unique(list(merge(left, right)))
        return merged

    def create_match_for_players(
        self, player_1: str, player_2: str, encoders, **kwargs
    ):
        match = {}
        for k in self.df.columns:
            if "winner" in k:
                match[k] = self.get_last_value_for_player(player_1, k)
            elif "loser" in k:
                match[k] = self.get_last_value_for_player(player_2, k)
            elif k in kwargs:
                match[k] = kwargs[k]
            else:
                match[k] = self.get_last_value_for_player(player_1, k)
        df = pd.DataFrame.from_records([match], index=[max(self.df.index) + 1])
        match_df = apply_encoders(df, encoders)
        df = pd.concat([self.df, match_df], axis=0)
        self.create_index(df)
        return match_df
