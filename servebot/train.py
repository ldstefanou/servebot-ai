import torch
import torch.nn.functional as F

# Read transformer model from yaml
from metrics import (
    log_loss,
    print_calibration_table,
    print_sequence_results,
    reset_metrics,
)
from model.servebot import TennisTransformer
from paths import save_model_artifacts


def validate(model: TennisTransformer, val_dl):
    model.eval()
    val_correct_total = 0
    val_seen_total = 0

    # Collect data for calibration analysis
    all_probas = []
    all_correct = []

    with torch.no_grad():
        for i, val_batch in enumerate(val_dl):
            out = model(val_batch)
            proba = out.softmax(-1)
            print_sequence_results(val_batch, out, model.embeddings)

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
    return val_accuracy_clean


def update_loss(out, batch):
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
    return loss


def train(tdl, vdl, config, embeddings):
    # Training loop
    reset_metrics()
    model = TennisTransformer(config, embeddings)

    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=config["epochs"] * len(tdl)
    )
    for ep in range(config["epochs"]):
        model.train()
        for i, batch in enumerate(tdl):
            out = model(batch)
            loss = update_loss(out, batch)
            log_loss(loss, out, batch, ep, i, len(tdl), scheduler.get_last_lr()[0])
            optim.step()
            scheduler.step()
            optim.zero_grad()
        acc = validate(model, vdl)
        # Save model after each epoch
        save_path = save_model_artifacts(
            model, config, f"epoch_{ep}", optimizer=optim, scheduler=scheduler
        )
        print(f"Model saved to {save_path}")
    return acc
