"""
TopicPulse v3 — PyTorch Dataset & DataLoader
Wraps preprocessed numpy arrays into a PyTorch Dataset for training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class QuestionQualityDataset(Dataset):
    """
    PyTorch Dataset for question quality prediction.

    Each sample returns:
        text_tensor        — (300,) int, padded text token indices
        code_tensor        — (200,) int, padded code token indices
        metadata_tensor    — (8,)   float, metadata features
        answer_label       — scalar float, 0/1 answerability
        time_label         — scalar int,   0-3 time bucket
    """

    def __init__(self, data_dict: dict):
        """
        Args:
            data_dict: dict with keys 'text', 'code', 'metadata',
                       'label_answer', 'label_time' — each a numpy array.
        """
        self.text = torch.from_numpy(data_dict["text"]).long()
        self.code = torch.from_numpy(data_dict["code"]).long()
        self.metadata = torch.from_numpy(data_dict["metadata"]).float()
        self.label_answer = torch.from_numpy(data_dict["label_answer"]).float()
        self.label_time = torch.from_numpy(data_dict["label_time"]).long()

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return (
            self.text[idx],
            self.code[idx],
            self.metadata[idx],
            self.label_answer[idx],
            self.label_time[idx],
        )


def create_dataloaders(splits: dict, batch_size: int = 64,
                       num_workers: int = 0) -> dict:
    """
    Create DataLoaders for train, val, and test splits.

    Args:
        splits: dict with 'train', 'val', 'test' keys, each a dict of
                numpy arrays.
        batch_size: batch size for training loader.
        num_workers: number of worker processes.

    Returns:
        dict with 'train', 'val', 'test' DataLoader objects.
    """
    loaders = {}

    for split_name, data in splits.items():
        ds = QuestionQualityDataset(data)
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split_name == "train"),
        )

    return loaders
