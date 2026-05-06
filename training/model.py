"""
TopicPulse v3 — QuestionQualityModel
Hybrid CNN-LSTM architecture for Stack Overflow question quality prediction.

Three branches:
    1. Text Branch:  Bidirectional LSTM on GloVe-initialized embeddings
    2. Code Branch:  Parallel multi-kernel Conv1d on learned embeddings
    3. Metadata Branch: Fully connected on 8 hand-crafted features

Two output heads:
    - Answerability (binary)  — will this question get an accepted answer?
    - Time bucket (4-class)   — how fast will the first answer arrive?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TextBranch(nn.Module):
    """Bidirectional LSTM over pretrained GloVe word embeddings."""

    def __init__(self, vocab_size=30000, embed_dim=100, hidden_dim=128,
                 num_layers=2, dropout=0.3, pretrained_weights=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weights))
            self.embedding.weight.requires_grad = True  # fine-tune

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # hidden_dim * 2 because bidirectional
        self.fc = nn.Linear(hidden_dim * 2, 128)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (batch, seq_len) int tensor of token indices."""
        emb = self.embedding(x)                           # (B, L, E)
        _, (h_n, _) = self.lstm(emb)                      # h_n: (layers*2, B, H)
        # Concatenate last forward and backward hidden states
        h_fwd = h_n[-2]                                   # (B, H)
        h_bwd = h_n[-1]                                   # (B, H)
        h_cat = torch.cat([h_fwd, h_bwd], dim=1)          # (B, 2H)
        out = self.dropout(F.relu(self.fc(h_cat)))         # (B, 128)
        return out


class CodeBranch(nn.Module):
    """Parallel multi-kernel Conv1d on learned code embeddings."""

    def __init__(self, vocab_size=10000, embed_dim=64, num_filters=64,
                 kernel_sizes=(3, 5, 7), dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters,
                      kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])

        total_filters = num_filters * len(kernel_sizes)  # 192
        self.fc = nn.Linear(total_filters, 64)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (batch, seq_len) int tensor of code token indices."""
        emb = self.embedding(x)              # (B, L, E)
        emb = emb.permute(0, 2, 1)          # (B, E, L)  — Conv1d expects (B, C, L)

        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))            # (B, F, L)
            p = F.adaptive_max_pool1d(c, 1)  # (B, F, 1)
            pooled.append(p.squeeze(-1))     # (B, F)

        cat = torch.cat(pooled, dim=1)       # (B, 192)
        out = self.dropout(F.relu(self.fc(cat)))  # (B, 64)
        return out


class MetadataBranch(nn.Module):
    """Fully connected branch for 8 hand-crafted metadata features."""

    def __init__(self, input_dim=8, output_dim=32, dropout=0.2):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (batch, 8) float tensor."""
        return self.dropout(F.relu(self.fc(x)))  # (B, 32)


class QuestionQualityModel(nn.Module):
    """
    Multi-branch model that fuses text, code, and metadata representations
    to predict question answerability (binary) and time-to-first-answer
    (4-class).
    """

    def __init__(self, text_vocab=30000, code_vocab=10000,
                 pretrained_text_weights=None):
        super().__init__()
        self.text_branch = TextBranch(
            vocab_size=text_vocab, pretrained_weights=pretrained_text_weights
        )
        self.code_branch = CodeBranch(vocab_size=code_vocab)
        self.metadata_branch = MetadataBranch()

        # Fusion layers: 128 + 64 + 32 = 224
        self.fusion_fc1 = nn.Linear(224, 128)
        self.fusion_bn = nn.BatchNorm1d(128)
        self.fusion_drop1 = nn.Dropout(0.4)

        self.fusion_fc2 = nn.Linear(128, 64)
        self.fusion_drop2 = nn.Dropout(0.3)

        # Output heads
        self.answerability_head = nn.Linear(64, 1)
        self.time_bucket_head = nn.Linear(64, 4)

    def forward(self, text_ids, code_ids, metadata):
        """
        Args:
            text_ids:  (B, 300) int tensor
            code_ids:  (B, 200) int tensor
            metadata:  (B, 8)  float tensor
        Returns:
            answerability: (B, 1) probabilities
            time_bucket:   (B, 4) logits (apply softmax externally if needed)
        """
        text_repr = self.text_branch(text_ids)       # (B, 128)
        code_repr = self.code_branch(code_ids)       # (B, 64)
        meta_repr = self.metadata_branch(metadata)   # (B, 32)

        fused = torch.cat([text_repr, code_repr, meta_repr], dim=1)  # (B, 224)

        x = F.relu(self.fusion_fc1(fused))           # (B, 128)
        x = self.fusion_bn(x)
        x = self.fusion_drop1(x)

        x = F.relu(self.fusion_fc2(x))               # (B, 64)
        x = self.fusion_drop2(x)

        answerability = torch.sigmoid(self.answerability_head(x))  # (B, 1)
        time_bucket = self.time_bucket_head(x)                     # (B, 4) raw logits

        return answerability, time_bucket
