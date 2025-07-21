# Trackers for running averages
from collections import deque

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate

from servebot.data.embeddings import decode_batch_of_embeddings

loss_tracker = deque(maxlen=100)
train_acc_tracker = deque(maxlen=100)
val_correct_tracker = deque(maxlen=100)
val_seen_tracker = deque(maxlen=100)


def log_loss(
    loss: torch.Tensor,
    out: torch.Tensor,
    batch: dict,
    ep: int,
    batch_no: int,
    total_batches: int,
    lr: float,
):
    # Calculate training and validation accuracy using existing predictions
    predictions = torch.argmax(out, dim=-1)
    correct = predictions == batch["target"]

    # Training accuracy (where is_validation == 0)
    train_correct = correct & batch["loss_mask"]
    train_accuracy = train_correct.sum() / batch["loss_mask"].sum()

    # Validation accuracy (where is_validation == 1)
    val_mask = batch["is_validation"]
    val_correct = correct & val_mask

    # Update trackers
    loss_tracker.append(loss.item())
    train_acc_tracker.append(train_accuracy.item())
    val_seen_tracker.append(val_mask.sum().item())
    val_correct_tracker.append(val_correct.sum().item())

    avg_loss = np.mean(loss_tracker)
    avg_train_acc = np.mean(train_acc_tracker)
    avg_val_acc = sum(val_correct_tracker) / (sum(val_seen_tracker) + 1)

    print(
        f"Epoch {ep}, Batch {batch_no}/{total_batches}: Avg Loss: {avg_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Val Acc: {avg_val_acc:.4f}, LR: {lr:.6f}"
    )


def print_sequence_results(batch: dict, out: torch.Tensor, mappings: dict):
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


def print_calibration_table(all_probas: list, all_correct: list):
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
