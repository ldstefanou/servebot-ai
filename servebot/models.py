from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from servebot.data.encodings import apply_encoders
from servebot.data.preprocess import preprocess_dataframe


@dataclass
class Match:
    winner_name: str
    loser_name: str

    # Match context with reasonable defaults
    surface: str = "Hard"
    tourney_name: str = "US Open"
    round: str = "R32"
    date: datetime = field(default_factory=datetime.now)

    # Player attributes
    winner_rank: int = 50
    loser_rank: int = 50
    winner_age: float = 25.0
    loser_age: float = 25.0
    winner_ht: float = 180
    loser_ht: float = 180
    winner_hand: str = "R"
    loser_hand: str = "R"
    winner_odds_avg: float = 1
    loser_odds_avg: float = 1

    # System fields (filled automatically)
    match_id: Optional[int] = None
    tourney_date: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary format for model input."""
        data = {
            "winner_name": self.winner_name,
            "loser_name": self.loser_name,
            "surface": self.surface,
            "tourney_name": self.tourney_name,
            "round": self.round,
            "date": self.date,
            "winner_rank": self.winner_rank,
            "loser_rank": self.loser_rank,
            "winner_age": self.winner_age,
            "loser_age": self.loser_age,
            "winner_ht": self.winner_ht,
            "loser_ht": self.loser_ht,
            "winner_hand": self.winner_hand,
            "loser_hand": self.loser_hand,
            "winner_odds_avg": self.winner_odds_avg,
            "loser_odds_avg": self.loser_odds_avg,
        }

        return data

    def standardize(self, encoders: Dict) -> "Match":
        """Standardize field values using fuzzy matching against encoders."""
        from servebot.data.utils import find_most_similar_string

        def fuzzy_match(value: str, encoder_key: str) -> str:
            if encoder_key not in encoders:
                return value
            valid_values = list(encoders[encoder_key].keys())
            return find_most_similar_string(valid_values, value)

        # Standardize categorical fields
        standardized = Match(
            winner_name=fuzzy_match(self.winner_name, "player_name"),
            loser_name=fuzzy_match(self.loser_name, "player_name"),
            surface=fuzzy_match(self.surface, "surface"),
            tourney_name=fuzzy_match(self.tourney_name, "tourney_name"),
            round=fuzzy_match(self.round, "round"),
            date=self.date,
            winner_rank=self.winner_rank,
            loser_rank=self.loser_rank,
            winner_age=self.winner_age,
            loser_age=self.loser_age,
            match_id=self.match_id,
            tourney_date=self.tourney_date,
        )

        return standardized

    @classmethod
    def from_df_record(cls, record):
        """Create Match from DataFrame record, ignoring unknown fields."""
        match_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in record.items() if k in match_fields}
        return cls(**filtered)

    @staticmethod
    def to_df(matches: List["Match"], encoders, starting_index=0):
        """Convert list of Match objects to DataFrame."""
        records = [match.standardize(encoders).to_dict() for match in matches]
        df = pd.DataFrame.from_records(records)
        df = preprocess_dataframe(df)
        df = apply_encoders(df, encoders)
        df.index = np.arange(starting_index, starting_index + len(df))
        return df
