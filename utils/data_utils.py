import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple


def negative_sample(n_items: int, exclude: set, rng=None) -> int:
    rng = rng or np.random.default_rng()
    while True:
        item = rng.integers(1, n_items)
        if item not in exclude:
            return int(item)


def build_sequence_dataset(
    data: List[Tuple],
    n_items: int,
    max_seq_len: int,
    user_sequences: Dict[int, List[int]],
    dataset_cls,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = dataset_cls(
        data=data,
        n_items=n_items,
        max_seq_len=max_seq_len,
        user_sequences=user_sequences,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )