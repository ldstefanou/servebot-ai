import argparse

import torch
import torch.nn.functional as F
import tqdm
from data.dataset import TennisDataset, ValSampler
from data.embeddings import create_embeddings
from data.preprocess import load_data, set_validation_matches

# Read transformer model from yaml
from metrics import log_loss, print_calibration_table, print_sequence_results
from model.servebot import TennisTransformer
from model.transformer import Transformer
from paths import save_model_artifacts
from torch.utils.data import DataLoader


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


def update_loss(out, batch, optim, scheduler):
    losses = F.cross_entropy(
        out.reshape(-1, out.shape[-1]),
        batch["target"].reshape(-1),
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

    args = parser.parse_args()

    # Create model
    base_model = Transformer.from_yaml("model_config.yaml")

    # Select dataset file
    df = load_data(sample=args.sample, filter_matches=base_model.config["dataset"])
    embeddings = create_embeddings(df, base_model.config["embedders"])
    df = set_validation_matches(df)

    # Create train dataset
    train_dataset = TennisDataset(
        embeddings=embeddings,
        df=df,
        config=base_model.config,
    )

    val_sampler = ValSampler(train_dataset)

    # Create dataloaders
    dl = DataLoader(
        train_dataset,
        batch_size=base_model.config["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    val_dataloader = DataLoader(
        train_dataset,
        batch_size=base_model.config["batch_size"],
        sampler=val_sampler,
        num_workers=0,
    )

    model = TennisTransformer(
        config=base_model.config,
        embeddings=embeddings,
    )
    # Training loop
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs * len(dl)
    )

    # Calculate total steps for progress bar
    total_steps = args.epochs * len(dl)

    with tqdm.tqdm(total=total_steps, desc="Training Progress") as pbar:
        for ep in range(args.epochs):
            model.train()
            for i, batch in enumerate(dl):
                out = model(batch)
                loss = update_loss(out, batch, optim, scheduler)
                log_loss(loss, out, batch, ep, i, len(dl), scheduler.get_last_lr()[0])

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "Epoch": f"{ep+1}/{args.epochs}",
                        "Batch": f"{i+1}/{len(dl)}",
                        "Loss": f"{loss.item():.4f}",
                        "LR": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )

            validate(model, val_dataloader)

            # Save model after each epoch
            save_path = save_model_artifacts(train_dataset, model, f"epoch_{ep}")
            print(f"Model saved to {save_path}")
