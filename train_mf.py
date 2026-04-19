"""
Train Matrix Factorization (BPR-MF) on Amazon dataset.

Usage:
    python train_mf.py --config configs/mf_config.yaml
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from data.amazon_dataset import AmazonDataset
from data.preprocessor import DataPreprocessor, PairwiseDataset
from models.matrix_factorization import MatrixFactorization
from trainers.mf_trainer import MFTrainer


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── data ──────────────────────────────────────────────────────────
    ds = AmazonDataset(
        category=cfg["dataset"]["category"],
        data_dir=cfg["dataset"]["data_dir"],
        min_interactions=cfg["dataset"]["min_interactions"],
    )
    df = ds.load()

    prep = DataPreprocessor(df, max_seq_len=cfg["dataset"]["max_seq_len"])
    prep.split()

    train_dataset = PairwiseDataset(
        data=prep.train_data,
        n_items=ds.n_items,
        user_sequences=prep.user_sequences,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # ── model ─────────────────────────────────────────────────────────
    model = MatrixFactorization(
        n_users=ds.n_users,
        n_items=ds.n_items,
        emb_dim=cfg["model"]["emb_dim"],
        dropout=cfg["model"]["dropout"],
    )
    print(f"MF parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── train ─────────────────────────────────────────────────────────
    trainer_cfg = {**cfg["training"], "max_seq_len": cfg["dataset"]["max_seq_len"]}
    trainer = MFTrainer(model, trainer_cfg, device)
    trainer.fit(train_loader, prep.val_data, prep.user_sequences, ds.n_items)

    # ── test ──────────────────────────────────────────────────────────
    trainer.load_best()
    test_metrics = trainer.evaluate(prep.test_data, prep.user_sequences, ds.n_items)
    print("\n=== Test Results ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mf_config.yaml")
    args = parser.parse_args()
    main(args.config)