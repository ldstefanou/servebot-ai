import argparse
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from data.dataset import TennisDataset, ValSampler
from data.utils import print_calibration_table, print_sequence_results

# Read transformer model from yaml
from model.servebot import TennisTransformer
from model.transformer import TransformerLLM
from paths import get_dataset_path, save_model_artifacts
from torch.utils.data import DataLoader

# Trackers for running averages
loss_tracker = deque(maxlen=100)
train_acc_tracker = deque(maxlen=100)
val_correct_tracker = deque(maxlen=100)
val_seen_tracker = deque(maxlen=100)


def validate(model, val_dl):
    val_correct_total = 0
    val_seen_total = 0

    # Collect data for calibration analysis
    all_probas = []
    all_correct = []

    with torch.no_grad():
        for i, val_batch in enumerate(val_dl):
            out = model(val_batch)
            proba = out.softmax(-1)
            print_sequence_results(val_batch, out, train_dataset.embeddings)

            val_predictions = torch.argmax(out, dim=-1)
            val_targets = val_batch["target"]
            val_mask = val_batch["is_validation"]

            val_correct = (val_predictions == val_targets) & val_mask
            val_correct_total += val_correct.sum().item()
            val_seen_total += val_mask.sum().item()

            # Collect calibration data (only for validation positions)
            max_probas = proba.max(dim=-1)[0]  # Confidence scores
            is_correct = val_correct

            # Only keep validation positions
            val_probas = max_probas[val_mask]
            val_is_correct = is_correct[val_mask]

            all_probas.extend(val_probas.cpu().numpy())
            all_correct.extend(val_is_correct.cpu().numpy())

    val_accuracy_clean = (
        val_correct_total / val_seen_total if val_seen_total > 0 else 0.0
    )
    print(f"Clean Validation Accuracy: {val_accuracy_clean:.4f}")

    print_calibration_table(all_probas, all_correct)
    print("=" * 80)


def log_loss(loss, out, batch, ep: int, batch_no: int, total_batches: int, lr: float):
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


def update_loss(out, batch, optim, scheduler):
    losses = F.cross_entropy(
        out.reshape(-1, out.shape[-1]),
        batch["target"].reshape(-1),
        ignore_index=0,
        reduction="none",
    ).reshape(
        out.shape[0], -1
    )  # Reshape back to [batch, seq]

    weighted_losses = losses * batch["loss_mask"]
    loss = weighted_losses.sum() / batch["loss_mask"].sum()
    loss.backward()
    optim.step()
    scheduler.step()
    optim.zero_grad()
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tennis prediction training")
    parser.add_argument(
        "--sample", type=int, default=None, help="Use only N samples for dry run"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "atp"],
        help="Dataset: all (all_levels) or atp (combined)",
    )
    parser.add_argument(
        "--sequence-type",
        type=str,
        default="player",
        choices=["player", "temporal", "match"],
        help="Sequence structure: player (player-specific), temporal (chronological sliding window), or match (combined player histories per match)",
    )
    args = parser.parse_args()

    # Create model
    base_model = TransformerLLM.from_yaml("model_config.yaml")

    # Select dataset file
    dataset_path = get_dataset_path(args.dataset)
    print(f"Using dataset: {dataset_path}")

    # Create train dataset
    train_dataset = TennisDataset(
        config_path="model_config.yaml",
        seq_length=base_model.config["seq_length"],
        sample_size=args.sample,
        dataset_path=dataset_path,
        sequence_type=args.sequence_type,
    )
    print(f"Using sequence type: {args.sequence_type}")

    val_sampler = ValSampler(train_dataset)

    # Create dataloaders
    dl = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=0,
    )

    model = TennisTransformer(
        config=base_model.config,
        embeddings=train_dataset.embeddings,
        target_size=train_dataset.target_size,
    )
    # Training loop
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs * len(dl)
    )

    for ep in range(args.epochs):
        model.train()
        for i, batch in enumerate(dl):
            out = model(batch)
            loss = update_loss(out, batch, optim, scheduler)
            log_loss(loss, out, batch, ep, i, len(dl), scheduler.get_last_lr()[0])

        validate(model, val_dataloader)

        # Save model after each epoch
        save_path = save_model_artifacts(train_dataset, model, f"epoch_{ep}")
        print(f"Model saved to {save_path}")
