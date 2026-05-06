"""
TopicPulse v3 — Ablation Studies
LSTM-only (no CNN, no metadata) and CNN-only (no LSTM, no metadata).
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.model import TextBranch, CodeBranch
from training.preprocess import preprocess_dataset, generate_synthetic_dataset, load_stackoverflow_csv
from training.dataset import create_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class LSTMOnlyModel(nn.Module):
    """Ablation: text branch only, no code or metadata."""
    def __init__(self, vocab_size=30000):
        super().__init__()
        self.text_branch = TextBranch(vocab_size=vocab_size)
        self.fc = nn.Linear(128, 64)
        self.head = nn.Linear(64, 1)

    def forward(self, text_ids, code_ids, metadata):
        x = self.text_branch(text_ids)
        x = F.relu(self.fc(x))
        return torch.sigmoid(self.head(x)), torch.zeros(text_ids.size(0), 4, device=text_ids.device)


class CNNOnlyModel(nn.Module):
    """Ablation: code branch only, no text or metadata."""
    def __init__(self, vocab_size=10000):
        super().__init__()
        self.code_branch = CodeBranch(vocab_size=vocab_size)
        self.fc = nn.Linear(64, 32)
        self.head = nn.Linear(32, 1)

    def forward(self, text_ids, code_ids, metadata):
        x = self.code_branch(code_ids)
        x = F.relu(self.fc(x))
        return torch.sigmoid(self.head(x)), torch.zeros(code_ids.size(0), 4, device=code_ids.device)


def train_ablation(model, loaders, name, device, epochs=15):
    """Train an ablation model and return test F1."""
    bce_fn = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for text, code, meta, y_ans, y_time in loaders["train"]:
            text, code, meta = text.to(device), code.to(device), meta.to(device)
            y_ans = y_ans.to(device)
            optimizer.zero_grad()
            ans_out, _ = model(text, code, meta)
            loss = bce_fn(ans_out.squeeze(1), y_ans)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for text, code, meta, y_ans, _ in loaders["val"]:
                text, code, meta = text.to(device), code.to(device), meta.to(device)
                ans_out, _ = model(text, code, meta)
                preds.extend((ans_out.squeeze(1) > 0.5).cpu().numpy())
                labels.extend(y_ans.numpy())
        f1 = f1_score(labels, preds, zero_division=0)
        best_f1 = max(best_f1, f1)
        logger.info("[%s] Epoch %2d — Val F1: %.4f", name, epoch, f1)

    # Test
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for text, code, meta, y_ans, _ in loaders["test"]:
            text, code, meta = text.to(device), code.to(device), meta.to(device)
            ans_out, _ = model(text, code, meta)
            preds.extend((ans_out.squeeze(1) > 0.5).cpu().numpy())
            labels.extend(y_ans.numpy())
    test_f1 = f1_score(labels, preds, zero_division=0)
    return test_f1


def run_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = os.environ.get("SO_DATA_CSV", "")
    if csv_path and os.path.exists(csv_path):
        df = load_stackoverflow_csv(csv_path)
    else:
        df = generate_synthetic_dataset(5000)

    splits, text_vocab, code_vocab = preprocess_dataset(df, output_dir="models")
    loaders = create_dataloaders(splits, batch_size=64)

    # LSTM-only
    lstm_model = LSTMOnlyModel(vocab_size=len(text_vocab)).to(device)
    lstm_f1 = train_ablation(lstm_model, loaders, "LSTM-only", device)

    # CNN-only
    cnn_model = CNNOnlyModel(vocab_size=len(code_vocab)).to(device)
    cnn_f1 = train_ablation(cnn_model, loaders, "CNN-only", device)

    logger.info("\n" + "=" * 50)
    logger.info("ABLATION RESULTS")
    logger.info("=" * 50)
    logger.info("%-15s  Test F1", "Model")
    logger.info("%-15s  %.4f", "LSTM-only", lstm_f1)
    logger.info("%-15s  %.4f", "CNN-only", cnn_f1)
    logger.info("=" * 50)


if __name__ == "__main__":
    run_ablation()
