"""
TopicPulse v3 — Model Evaluation
Compute metrics, generate confusion matrix, and training curves.
"""

import os
import sys
import logging
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.model import QuestionQualityModel
from training.preprocess import preprocess_dataset, generate_synthetic_dataset, load_stackoverflow_csv
from training.dataset import create_dataloaders

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TIME_BUCKET_LABELS = ["under_1h", "1h_to_24h", "1d_to_7d", "over_7d_or_never"]


def evaluate_model(model_path: str = "models/question_quality.pt",
                   output_dir: str = "models/eval"):
    """Full model evaluation on test set."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    csv_path = os.environ.get("SO_DATA_CSV", "")
    if csv_path and os.path.exists(csv_path):
        df = load_stackoverflow_csv(csv_path)
    else:
        logger.info("No CSV found — generating synthetic data for evaluation.")
        df = generate_synthetic_dataset(5000)

    splits, text_vocab, code_vocab = preprocess_dataset(df, output_dir="models")
    loaders = create_dataloaders(splits, batch_size=64)

    # Load model
    model = QuestionQualityModel(
        text_vocab=len(text_vocab),
        code_vocab=len(code_vocab),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Collect predictions
    all_ans_probs = []
    all_ans_labels = []
    all_time_preds = []
    all_time_labels = []

    with torch.no_grad():
        for text, code, meta, y_ans, y_time in loaders["test"]:
            text, code, meta = text.to(device), code.to(device), meta.to(device)
            ans_out, time_out = model(text, code, meta)

            all_ans_probs.extend(ans_out.squeeze(1).cpu().numpy())
            all_ans_labels.extend(y_ans.numpy())
            all_time_preds.extend(time_out.argmax(dim=1).cpu().numpy())
            all_time_labels.extend(y_time.numpy())

    ans_probs = np.array(all_ans_probs)
    ans_labels = np.array(all_ans_labels)
    ans_preds = (ans_probs > 0.5).astype(int)
    time_preds = np.array(all_time_preds)
    time_labels = np.array(all_time_labels)

    # ------------------------------------------------------------------
    # Binary answerability metrics
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 50)
    logger.info("BINARY ANSWERABILITY METRICS")
    logger.info("=" * 50)
    acc = accuracy_score(ans_labels, ans_preds)
    prec = precision_score(ans_labels, ans_preds, zero_division=0)
    rec = recall_score(ans_labels, ans_preds, zero_division=0)
    f1 = f1_score(ans_labels, ans_preds, zero_division=0)
    try:
        auc = roc_auc_score(ans_labels, ans_probs)
    except ValueError:
        auc = 0.0

    logger.info("Accuracy:  %.4f", acc)
    logger.info("Precision: %.4f", prec)
    logger.info("Recall:    %.4f", rec)
    logger.info("F1 Score:  %.4f", f1)
    logger.info("ROC-AUC:   %.4f", auc)

    # ------------------------------------------------------------------
    # Time bucket metrics
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 50)
    logger.info("TIME BUCKET METRICS")
    logger.info("=" * 50)
    macro_f1 = f1_score(time_labels, time_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(time_labels, time_preds, average="weighted", zero_division=0)
    logger.info("Macro F1:    %.4f", macro_f1)
    logger.info("Weighted F1: %.4f", weighted_f1)
    logger.info("\nPer-class report:\n%s",
                classification_report(time_labels, time_preds,
                                      target_names=TIME_BUCKET_LABELS,
                                      zero_division=0))

    # ------------------------------------------------------------------
    # Confusion matrices
    # ------------------------------------------------------------------
    # Answerability confusion matrix
    cm_ans = confusion_matrix(ans_labels, ans_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_ans, interpolation="nearest", cmap="Blues")
    ax.set_title("Answerability Confusion Matrix", fontsize=13)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_ans[i, j]), ha="center", va="center",
                    color="white" if cm_ans[i, j] > cm_ans.max() / 2 else "black",
                    fontsize=14)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_answerability.png"), dpi=150)
    plt.close()

    # Time bucket confusion matrix
    cm_time = confusion_matrix(time_labels, time_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_time, interpolation="nearest", cmap="Oranges")
    ax.set_title("Time Bucket Confusion Matrix", fontsize=13)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    n_classes = len(TIME_BUCKET_LABELS)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(TIME_BUCKET_LABELS, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(TIME_BUCKET_LABELS, fontsize=9)
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm_time[i, j] if i < cm_time.shape[0] and j < cm_time.shape[1] else 0
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if val > cm_time.max() / 2 else "black",
                    fontsize=12)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_time_bucket.png"), dpi=150)
    plt.close()

    logger.info("Confusion matrices saved to %s", output_dir)

    # ------------------------------------------------------------------
    # Save metrics to JSON
    # ------------------------------------------------------------------
    metrics = {
        "answerability": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(auc, 4),
        },
        "time_bucket": {
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
        },
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    evaluate_model()
