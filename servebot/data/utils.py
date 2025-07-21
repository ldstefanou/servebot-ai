from collections import defaultdict
from heapq import merge

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from data.embeddings import decode_batch_of_embeddings
from tabulate import tabulate
from torch.nn.utils.rnn import pad_sequence


def masked_softmax(logits, mask):
    """
    Apply softmax only over masked positions.

    Args:
        logits: Tensor of shape [..., vocab_size]
        mask: Boolean tensor of shape [..., vocab_size] where True means valid position

    Returns:
        Probabilities tensor of same shape as logits
    """
    # Set masked positions to very negative value
    masked_logits = torch.where(mask, logits, torch.full_like(logits, -float("inf")))
    # Apply softmax - exp(-inf) = 0, so masked positions get 0 probability
    return F.softmax(masked_logits, dim=-1)


def merge_dicts_sorted(
    d1: dict[str, list[int]], d2: dict[str, list[int]]
) -> dict[str, list[int]]:
    result = defaultdict(list)
    all_keys = set(d1) | set(d2)

    for key in all_keys:
        list1 = d1.get(key, [])
        list2 = d2.get(key, [])
        # merge keeps the result sorted if inputs are sorted
        result[key] = list(merge(list1, list2))

    return dict(result)


def truncate(sequence, length):
    seqs = [sequence[i : i + length] for i in range(0, len(sequence), length)]
    return seqs


def truncate_and_pad_to_long_tensor(
    list_of_sequences, sequence_length, padding_value=0
):
    trunced = [truncate(s, sequence_length) for s in list_of_sequences]
    list_of_pt = [torch.tensor(y) for r in trunced for y in r]
    return pad_sequence(list_of_pt, batch_first=True, padding_value=padding_value)


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def print_sequence_results(batch: dict, out: torch.tensor, mappings):
    # random sequence from batch
    for rnd in range(out.shape[0]):

        sequence_length = (~batch["is_padding"]).cumsum(dim=1).argmax(-1)[rnd] + 1
        decoded = decode_batch_of_embeddings(batch, mappings)

        left_p = decoded["winner_name_token"][rnd, :sequence_length].tolist()
        right_p = decoded["loser_name_token"][rnd, :sequence_length].tolist()
        left_rank = decoded["winner_rank_token"][rnd, :sequence_length].tolist()
        right_rank = decoded["loser_rank_token"][rnd, :sequence_length].tolist()
        left_age = batch["winner_age"][rnd, :sequence_length].tolist()
        right_age = batch["loser_age"][rnd, :sequence_length].tolist()
        tournament = decoded["tournament_token"][rnd, :sequence_length].tolist()
        year = decoded["year_token"][rnd, :sequence_length].tolist()

        targets = batch["target"][rnd, :sequence_length].tolist()
        confidence, preds = out.softmax(-1).max(dim=-1)
        confidence = confidence[rnd, :sequence_length].tolist()
        preds = preds[rnd, :sequence_length].tolist()
        is_validation = batch["is_validation"][rnd, :sequence_length].tolist()

        for i in range(sequence_length):
            is_val_match = is_validation[i]
            if not is_val_match:
                continue
            pred = preds[i]
            target = targets[i]
            # Emojis: differentiate validation vs training context
            correct_emoji = "✅" if pred == target else "❌"
            if is_val_match:
                match_emoji = "🏆"  # Validation match (being predicted)
            else:
                match_emoji = "📝"  # Training context match (provides context)

            pred_winner = "L" if pred == 2 else "R"
            true_winner = "L" if target == 2 else "R"

            left_player = f"{match_emoji} {left_p[i]}-{left_rank[i]} ({left_age[i]})"
            right_player = f"{right_p[i]}-{right_rank[i]} ({right_age[i]})"
            event_info = f"{tournament[i]} {year[i]}"
            pred_info = f"Pred: {pred}({pred_winner}), True: {target}({true_winner}), Conf: {confidence[i]:.3f} {correct_emoji}"
            print(f"{left_player} vs {right_player} | {event_info} | {pred_info}")


def print_calibration_table(all_probas, all_correct):
    df_cal = pd.DataFrame({"confidence": all_probas, "is_correct": all_correct})

    # Create confidence bins
    bins = np.linspace(0, 1, 11)  # 10 bins: [0-0.1), [0.1-0.2), ..., [0.9-1.0]
    df_cal["confidence_bin"] = pd.cut(
        df_cal["confidence"], bins=bins, include_lowest=True
    )

    # Calculate calibration metrics per bin
    cal_stats = (
        df_cal.groupby("confidence_bin")
        .agg({"is_correct": ["mean", "count"], "confidence": "mean"})
        .round(3)
    )

    cal_stats.columns = ["Accuracy", "Count", "Avg_Confidence"]
    cal_stats = cal_stats.reset_index()
    cal_stats["Bin"] = [
        f"{interval.left:.1f}-{interval.right:.1f}"
        for interval in cal_stats["confidence_bin"]
    ]

    print("\n" + "=" * 50)
    print("MODEL CALIBRATION ANALYSIS")
    print("=" * 50)
    print(
        tabulate(
            cal_stats[["Bin", "Avg_Confidence", "Accuracy", "Count"]].values,
            headers=["Confidence Bin", "Avg Confidence", "Accuracy", "Count"],
            tablefmt="grid",
        )
    )

    # Overall calibration error
    weighted_bins = cal_stats.dropna()
    cal_error = np.average(
        np.abs(weighted_bins["Avg_Confidence"] - weighted_bins["Accuracy"]),
        weights=weighted_bins["Count"],
    )
    print(f"\nExpected Calibration Error: {cal_error:.4f}")
