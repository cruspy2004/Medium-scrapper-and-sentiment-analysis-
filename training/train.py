"""
TopicPulse v3 — Training Loop
Trains the QuestionQualityModel with early stopping, LR scheduling,
gradient clipping, and best-model checkpointing.
"""

import os
import sys
import time
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score

# Add project root to path so we can import training modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.model import QuestionQualityModel
from training.preprocess import preprocess_dataset, generate_synthetic_dataset, load_stackoverflow_csv
from training.dataset import create_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ===================================================================
# CONFIGURATION
# ===================================================================
CONFIG = {
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 64,
    "max_epochs": 30,
    "patience": 5,           # early stopping on val F1
    "lr_patience": 3,        # ReduceLROnPlateau
    "lr_factor": 0.5,
    "max_grad_norm": 1.0,
    "bce_weight": 0.7,
    "ce_weight": 0.3,
    "model_dir": "models",
}


# ===================================================================
# TRAINING
# ===================================================================

def compute_class_weights(labels: np.ndarray, n_classes: int = 4) -> torch.Tensor:
    """Compute inverse-frequency class weights for time bucket CE loss."""
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes  # normalize
    return torch.FloatTensor(weights)


def evaluate(model, loader, bce_fn, ce_fn, device):
    """Evaluate model on a data loader. Returns loss, f1, accuracy."""
    model.eval()
    total_loss = 0.0
    all_preds_ans = []
    all_labels_ans = []
    all_preds_time = []
    all_labels_time = []
    n_batches = 0

    with torch.no_grad():
        for text, code, meta, y_ans, y_time in loader:
            text = text.to(device)
            code = code.to(device)
            meta = meta.to(device)
            y_ans = y_ans.to(device)
            y_time = y_time.to(device)

            ans_out, time_out = model(text, code, meta)

            loss_bce = bce_fn(ans_out.squeeze(1), y_ans)
            loss_ce = ce_fn(time_out, y_time)
            loss = CONFIG["bce_weight"] * loss_bce + CONFIG["ce_weight"] * loss_ce
            total_loss += loss.item()
            n_batches += 1

            # Predictions
            preds_ans = (ans_out.squeeze(1) > 0.5).cpu().numpy()
            all_preds_ans.extend(preds_ans)
            all_labels_ans.extend(y_ans.cpu().numpy())

            preds_time = time_out.argmax(dim=1).cpu().numpy()
            all_preds_time.extend(preds_time)
            all_labels_time.extend(y_time.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    f1_ans = f1_score(all_labels_ans, all_preds_ans, average="binary", zero_division=0)
    acc_ans = np.mean(np.array(all_labels_ans) == np.array(all_preds_ans))

    return avg_loss, f1_ans, acc_ans


def train():
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Data loading & preprocessing
    # ------------------------------------------------------------------
    csv_path = os.environ.get("SO_DATA_CSV", "")
    if csv_path and os.path.exists(csv_path):
        df = load_stackoverflow_csv(csv_path)
    else:
        logger.info("No SO_DATA_CSV set — generating synthetic dataset.")
        os.makedirs("data", exist_ok=True)
        df = generate_synthetic_dataset(5000, "data/synthetic_so.csv")

    splits, text_vocab, code_vocab = preprocess_dataset(df, output_dir=CONFIG["model_dir"])
    loaders = create_dataloaders(splits, batch_size=CONFIG["batch_size"])

    # ------------------------------------------------------------------
    # Model, loss, optimizer
    # ------------------------------------------------------------------
    model = QuestionQualityModel(
        text_vocab=len(text_vocab),
        code_vocab=len(code_vocab),
    ).to(device)

    logger.info("Model parameters: %s",
                f"{sum(p.numel() for p in model.parameters()):,}")

    bce_fn = nn.BCELoss()
    class_weights = compute_class_weights(splits["train"]["label_time"])
    ce_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = Adam(model.parameters(), lr=CONFIG["lr"],
                     weight_decay=CONFIG["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="max",
                                  patience=CONFIG["lr_patience"],
                                  factor=CONFIG["lr_factor"],
                                  verbose=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_f1 = 0.0
    epochs_no_improve = 0
    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    model_path = os.path.join(CONFIG["model_dir"], "question_quality.pt")

    logger.info("=" * 70)
    logger.info("Starting training — max %d epochs, patience %d",
                CONFIG["max_epochs"], CONFIG["patience"])
    logger.info("=" * 70)

    for epoch in range(1, CONFIG["max_epochs"] + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0

        for text, code, meta, y_ans, y_time in loaders["train"]:
            text = text.to(device)
            code = code.to(device)
            meta = meta.to(device)
            y_ans = y_ans.to(device)
            y_time = y_time.to(device)

            optimizer.zero_grad()
            ans_out, time_out = model(text, code, meta)

            loss_bce = bce_fn(ans_out.squeeze(1), y_ans)
            loss_ce = ce_fn(time_out, y_time)
            loss = CONFIG["bce_weight"] * loss_bce + CONFIG["ce_weight"] * loss_ce

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        val_loss, val_f1, val_acc = evaluate(model, loaders["val"], bce_fn, ce_fn, device)

        scheduler.step(val_f1)
        elapsed = time.time() - epoch_start

        logger.info(
            "Epoch %2d/%d | Train Loss: %.4f | Val Loss: %.4f | "
            "Val F1: %.4f | Val Acc: %.4f | Time: %.1fs | LR: %.2e",
            epoch, CONFIG["max_epochs"], train_loss, val_loss,
            val_f1, val_acc, elapsed,
            optimizer.param_groups[0]["lr"]
        )

        # Checkpointing
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            logger.info("  → New best model saved (F1: %.4f)", val_f1)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CONFIG["patience"]:
                logger.info("Early stopping triggered after %d epochs.", epoch)
                break

    logger.info("=" * 70)
    logger.info("Training complete. Best validation F1: %.4f", best_val_f1)
    logger.info("Model saved to: %s", model_path)
    logger.info("=" * 70)

    return model_path


if __name__ == "__main__":
    train()
