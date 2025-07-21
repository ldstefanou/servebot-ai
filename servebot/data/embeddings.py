from typing import Dict, List

import numpy as np
import pandas as pd


def create_embeddings(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, int]]:
    embeddings = {}
    for col in cols:
        series = df[col]
        if col.startswith("winner_"):
            column_name = col.split("winner_")[-1]
            loser_side = f"loser_{column_name}"
            series = pd.concat([df[col], df[loser_side]]).rename(col)
            col = f"player_{column_name}"

        array = series.astype("category").cat.codes + 1
        mapping = array.groupby(series).first().to_dict()
        mapping[f"{col}_padding"] = 0
        embeddings[col] = mapping

    return embeddings


def apply_embeddings(df: pd.DataFrame, embeddings: Dict[str, Dict[str, int]]):
    for embedding_key, mapping in embeddings.items():
        if embedding_key.startswith("player_"):
            # Shared player embeddings - apply to both winner and loser
            feature = embedding_key.removeprefix("player_")
            df[f"winner_{feature}_token"] = df[f"winner_{feature}"].map(mapping)
            df[f"loser_{feature}_token"] = df[f"loser_{feature}"].map(mapping)
        else:
            # Regular embeddings
            df[f"{embedding_key}_token"] = df[embedding_key].map(mapping).fillna(0)

    # Keep only rows with complete embedding data
    token_columns = [col for col in df.columns if col.endswith("_token")]
    df[token_columns] = (
        df[token_columns].fillna(0).astype({k: "int" for k in token_columns})
    )
    return df.dropna(subset=token_columns).reset_index(drop=True)


def decode_batch_of_embeddings(batch, embeddings) -> Dict:
    reverse_embeddings = {}
    for key, mapping in embeddings.items():
        reverse_embeddings[key] = {v: k for k, v in mapping.items()}

    decoded = {}
    for key, tensor in batch.items():
        b, s = tensor.shape
        if key.endswith("_token"):
            embedding_key = key.replace("_token", "")
            if embedding_key.startswith("winner_") or embedding_key.startswith(
                "loser_"
            ):
                base_key = embedding_key.split("_", 1)[1]
                embedding_key = f"player_{base_key}"

            if embedding_key in reverse_embeddings:
                mapping = reverse_embeddings[embedding_key]
                vals = [mapping[t.item()] for t in tensor.flatten()]
                decoded[key] = np.asarray(vals).reshape(b, s)

    return decoded
