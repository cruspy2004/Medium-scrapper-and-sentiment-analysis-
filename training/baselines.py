"""
TopicPulse v3 — Baseline Models
TF-IDF + LR (text) and Random Forest (metadata) for comparison.
"""

import os
import sys
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.preprocess import preprocess_dataset, generate_synthetic_dataset, load_stackoverflow_csv, strip_html

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_baselines():
    csv_path = os.environ.get("SO_DATA_CSV", "")
    if csv_path and os.path.exists(csv_path):
        df = load_stackoverflow_csv(csv_path)
    else:
        logger.info("No CSV — using synthetic data.")
        df = generate_synthetic_dataset(5000)

    splits, _, _ = preprocess_dataset(df, output_dir="models")

    # --- Baseline 1: TF-IDF + Logistic Regression ---
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE 1: TF-IDF + Logistic Regression (text only)")
    logger.info("=" * 60)

    df["body_text"] = df["Body"].apply(strip_html)
    df["full_text"] = df["Title"].fillna("") + " " + df["body_text"]
    labels = df["AcceptedAnswerId"].notna().astype(int).values

    train_t, temp_t, train_l, temp_l = train_test_split(
        df["full_text"].values, labels, test_size=0.30, random_state=42, stratify=labels)
    _, test_t, _, test_l = train_test_split(
        temp_t, temp_l, test_size=0.50, random_state=42, stratify=temp_l)

    tfidf = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 2))
    X_train = tfidf.fit_transform(train_t)
    X_test = tfidf.transform(test_t)

    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_train, train_l)
    preds = lr.predict(X_test)
    probs = lr.predict_proba(X_test)[:, 1]

    _print_metrics("TF-IDF + LR", test_l, preds, probs)

    # --- Baseline 2: Random Forest (metadata only) ---
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE 2: Random Forest (metadata only)")
    logger.info("=" * 60)

    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(splits["train"]["metadata"], splits["train"]["label_answer"])
    preds = rf.predict(splits["test"]["metadata"])
    probs = rf.predict_proba(splits["test"]["metadata"])[:, 1]

    _print_metrics("Random Forest", splits["test"]["label_answer"], preds, probs)


def _print_metrics(name, labels, preds, probs):
    logger.info("Accuracy:  %.4f", accuracy_score(labels, preds))
    logger.info("Precision: %.4f", precision_score(labels, preds, zero_division=0))
    logger.info("Recall:    %.4f", recall_score(labels, preds, zero_division=0))
    logger.info("F1:        %.4f", f1_score(labels, preds, zero_division=0))
    try:
        logger.info("ROC-AUC:   %.4f", roc_auc_score(labels, probs))
    except ValueError:
        logger.info("ROC-AUC:   N/A")


if __name__ == "__main__":
    run_baselines()
