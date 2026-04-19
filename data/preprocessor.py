import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict


class DataPreprocessor:
    """
    Splits interactions into train / val / test using leave-one-out strategy.

    For each user the last item is test, second-to-last is val, rest is train.
    Also builds per-user item sequences (chronologically sorted).
    """

    def __init__(self, df: pd.DataFrame, max_seq_len: int = 50):
        self.df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        self.max_seq_len = max_seq_len
        self.n_users = df["user_id"].nunique()
        self.n_items = df["item_id"].nunique()

        self.user_sequences: Dict[int, List[int]] = {}
        self.train_data: List[Tuple] = []
        self.val_data: List[Tuple] = []
        self.test_data: List[Tuple] = []

    def split(self):
        grouped = self.df.groupby("user_id")["item_id"].apply(list)

        for user, items in grouped.items():
            if len(items) < 3:
                continue
            self.user_sequences[user] = items

            # leave-one-out
            test_item  = items[-1]
            val_item   = items[-2]
            train_items = items[:-2]

            for idx, item in enumerate(train_items):
                history = train_items[max(0, idx - self.max_seq_len): idx]
                self.train_data.append((user, item, history))

            val_history = train_items[-self.max_seq_len:]
            self.val_data.append((user, val_item, val_history))

            test_history = items[:-1][-self.max_seq_len:]
            self.test_data.append((user, test_item, test_history))

        print(f"[Preprocessor] train={len(self.train_data)} | val={len(self.val_data)} | test={len(self.test_data)}")
        return self

    def get_all_item_ids(self) -> List[int]:
        return list(range(self.n_items))


class SequenceDataset(Dataset):
    """
    Dataset for sequential / stateful models.
    Each sample: (user, target_item, history_sequence, negative_item)
    History is padded/truncated to max_seq_len.
    """

    def __init__(
        self,
        data: List[Tuple],
        n_items: int,
        max_seq_len: int = 50,
        n_neg: int = 1,
        user_sequences: Dict[int, List[int]] = None,
    ):
        self.data = data
        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.n_neg = n_neg
        self.user_sequences = user_sequences or {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user, pos_item, history = self.data[idx]

        # negative sampling
        user_items = set(self.user_sequences.get(user, []))
        neg_item = self._sample_negative(user_items)

        seq = self._pad(history)
        seq_len = min(len(history), self.max_seq_len)

        return {
            "user":     torch.tensor(user, dtype=torch.long),
            "pos_item": torch.tensor(pos_item, dtype=torch.long),
            "neg_item": torch.tensor(neg_item, dtype=torch.long),
            "seq":      torch.tensor(seq, dtype=torch.long),
            "seq_len":  torch.tensor(seq_len, dtype=torch.long),
        }

    def _pad(self, history: List[int]) -> List[int]:
        truncated = history[-self.max_seq_len:]
        padded = [0] * (self.max_seq_len - len(truncated)) + truncated
        return padded

    def _sample_negative(self, user_items: set) -> int:
        while True:
            neg = np.random.randint(1, self.n_items)
            if neg not in user_items:
                return neg


class PairwiseDataset(Dataset):
    """
    BPR-style (user, pos_item, neg_item) dataset — no sequence info.
    Used for MF and NCF trainers.
    """

    def __init__(self, data: List[Tuple], n_items: int, user_sequences: Dict[int, List[int]] = None):
        self.data = data
        self.n_items = n_items
        self.user_sequences = user_sequences or {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user, pos_item, _ = self.data[idx]
        user_items = set(self.user_sequences.get(user, []))
        neg_item = self._sample_negative(user_items)
        return {
            "user":     torch.tensor(user, dtype=torch.long),
            "pos_item": torch.tensor(pos_item, dtype=torch.long),
            "neg_item": torch.tensor(neg_item, dtype=torch.long),
        }

    def _sample_negative(self, user_items: set) -> int:
        while True:
            neg = np.random.randint(1, self.n_items)
            if neg not in user_items:
                return neg