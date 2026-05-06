"""
TopicPulse v3 — Question Quality Inference Pipeline
Loads trained CNN-LSTM model and runs batch inference on scraped SO questions.
"""

import os
import re
import json
import html
import logging

import torch
import numpy as np
from bs4 import BeautifulSoup

# Import model architecture
from training.model import QuestionQualityModel

logger = logging.getLogger(__name__)

# Preprocessing constants (must match training)
TEXT_MAX_LEN = 300
CODE_MAX_LEN = 200


class QuestionQualityPredictor:
    """
    Inference wrapper for the QuestionQualityModel.
    Loaded once at app startup and reused across all scan requests.
    """

    def __init__(self, model_path: str, vocab_text_path: str,
                 vocab_code_path: str):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load vocabularies
        with open(vocab_text_path) as f:
            self.vocab_text = json.load(f)
        with open(vocab_code_path) as f:
            self.vocab_code = json.load(f)

        # Load model
        self.model = QuestionQualityModel(
            text_vocab=len(self.vocab_text),
            code_vocab=len(self.vocab_code),
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        logger.info("QuestionQualityPredictor loaded on %s", self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_batch(self, questions: list) -> list:
        """
        Takes a list of scraped question dicts and adds quality predictions
        in place. Returns the same list with added fields:
            - answerability (float 0-1)
            - time_bucket (str)
            - quality_label (str: HIGH/MEDIUM/LOW)
        """
        if not questions:
            return questions

        text_tensors = self._encode_text(questions)
        code_tensors = self._encode_code(questions)
        meta_tensors = self._encode_metadata(questions)

        with torch.no_grad():
            answerability, time_bucket = self.model(
                text_tensors.to(self.device),
                code_tensors.to(self.device),
                meta_tensors.to(self.device),
            )

        bucket_labels = ["under_1h", "1h_to_24h", "1d_to_7d", "over_7d_or_never"]

        for i, q in enumerate(questions):
            prob = answerability[i].item()
            bucket = bucket_labels[time_bucket[i].argmax().item()]
            label = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"

            q["answerability"] = round(prob, 3)
            q["time_bucket"] = bucket
            q["quality_label"] = label

        return questions

    # ------------------------------------------------------------------
    # Encoding helpers (mirror training preprocessing)
    # ------------------------------------------------------------------

    def _encode_text(self, questions: list) -> torch.Tensor:
        """Tokenize and encode title + body text."""
        indices_batch = []
        for q in questions:
            title = q.get("title", "")
            body = q.get("body_text", "")
            text = f"{title} {body}".lower()

            # Simple tokenization (no NLTK dependency at runtime)
            tokens = re.findall(r"[a-z]+", text)
            ids = [
                self.vocab_text.get(t, self.vocab_text.get("<UNK>", 1))
                for t in tokens[:TEXT_MAX_LEN]
            ]
            ids += [0] * (TEXT_MAX_LEN - len(ids))
            indices_batch.append(ids)

        return torch.tensor(indices_batch, dtype=torch.long)

    def _encode_code(self, questions: list) -> torch.Tensor:
        """Tokenize and encode code blocks."""
        indices_batch = []
        for q in questions:
            code_blocks = q.get("code_blocks", [])
            code_text = " ".join(code_blocks) if code_blocks else ""

            tokens = re.findall(
                r"[a-zA-Z_]\w*|[+\-*/=<>!&|^~%]+|\d+", code_text
            )
            ids = [
                self.vocab_code.get(t.lower(), self.vocab_code.get("<UNK>", 1))
                for t in tokens[:CODE_MAX_LEN]
            ]
            ids += [0] * (CODE_MAX_LEN - len(ids))
            indices_batch.append(ids)

        return torch.tensor(indices_batch, dtype=torch.long)

    def _encode_metadata(self, questions: list) -> torch.Tensor:
        """Extract 8 metadata features per question."""
        meta_batch = []
        for q in questions:
            tags = q.get("tags", [])
            if isinstance(tags, str):
                tags = tags.split(",")
            title = q.get("title", "")
            body = q.get("body_text", "")
            code_blocks = q.get("code_blocks", [])
            code_text = " ".join(code_blocks) if code_blocks else ""

            meta = [
                len(tags),                              # tag_count
                len(title.split()),                     # title_length
                len(body.split()),                      # body_length
                len(code_text.split()),                 # code_length
                1.0 if code_blocks else 0.0,            # has_code
                len(code_blocks),                       # num_code_blocks
                12.0,                                   # hour_of_day (default)
                0.0,                                    # is_weekend (default)
            ]
            meta_batch.append(meta)

        return torch.tensor(meta_batch, dtype=torch.float32)
