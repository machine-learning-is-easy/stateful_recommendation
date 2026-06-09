"""
Train StatefulRecommender on Amazon dataset.

The StatefulRecommender wraps any base model (MF / NCF / SASRec) with a
pluggable UserStateEncoder that converts purchase history into a dynamic
user state embedding, replacing the static user embedding.

Usage:
    python train_stateful.py --config configs/stateful_config.yaml
    python train_stateful.py --base ncf
    python train_stateful.py --base sasrec --encoder mamba
    python train_stateful.py --encoder causal_transformer
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from data.amazon_dataset import AmazonDataset
from data.preprocessor import DataPreprocessor, SequenceDataset
from models.stateful_rec import StatefulRecommender
from trainers.stateful_trainer import StatefulTrainer


def main(config_path: str, base_override: str = None, encoder_override: str = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if base_override:
        cfg["model"]["base_model_type"] = base_override
    if encoder_override:
        cfg["model"]["encoder_type"] = encoder_override

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_type    = cfg["model"]["base_model_type"]
    encoder_type = cfg["model"].get("encoder_type", "gru")
    print(f"Device: {device} | Base model: {base_type} | Encoder: {encoder_type}")

    # ── data ──────────────────────────────────────────────────────────
    ds = AmazonDataset(
        category=cfg["dataset"]["category"],
        data_dir=cfg["dataset"]["data_dir"],
        min_interactions=cfg["dataset"]["min_interactions"],
    )
    df = ds.load()

    prep = DataPreprocessor(df, max_seq_len=cfg["dataset"]["max_seq_len"])
    prep.split()

    train_dataset = SequenceDataset(
        data=prep.train_data,
        n_items=ds.n_items,
        max_seq_len=cfg["dataset"]["max_seq_len"],
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
    model_cfg = cfg["model"]
    extra_kwargs = {}
    if base_type == "ncf":
        extra_kwargs["mlp_layers"] = model_cfg.get("mlp_layers", [256, 128, 64])
    if base_type == "sasrec":
        extra_kwargs["n_blocks"] = model_cfg.get("n_blocks", 2)
        extra_kwargs["n_heads"]  = model_cfg.get("n_heads", 2)

    model = StatefulRecommender(
        base_model_type=base_type,
        n_users=ds.n_users,
        n_items=ds.n_items,
        emb_dim=model_cfg["emb_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        n_layers=model_cfg.get("n_layers", model_cfg.get("gru_layers", 2)),
        max_seq_len=cfg["dataset"]["max_seq_len"],
        dropout=model_cfg["dropout"],
        encoder_type=encoder_type,
        bidirectional=model_cfg.get("bidirectional", False),
        d_state=model_cfg.get("d_state", 16),
        enc_n_heads=model_cfg.get("enc_n_heads", 4),
        **extra_kwargs,
    )
    total_params   = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"StatefulRecommender({base_type} + {encoder_type}) parameters: {total_params:,}")
    print(f"  |-- UserStateEncoder ({encoder_type}): {encoder_params:,}")
    print(f"  |-- Base model ({base_type}):           {total_params - encoder_params:,}")

    # ── train ─────────────────────────────────────────────────────────
    run_tag   = f"{base_type}_{encoder_type}"
    ckpt_path = cfg["training"]["checkpoint_path"].replace("stateful_best", f"stateful_{run_tag}_best")
    log_dir   = cfg["training"]["log_dir"] + f"_{run_tag}"
    trainer_cfg = {
        **cfg["training"],
        "checkpoint_path": ckpt_path,
        "log_dir": log_dir,
        "max_seq_len": cfg["dataset"]["max_seq_len"],
    }
    trainer = StatefulTrainer(model, trainer_cfg, device)
    trainer.fit(train_loader, prep.val_data, prep.user_sequences, ds.n_items)

    # ── test ──────────────────────────────────────────────────────────
    trainer.load_best()
    test_metrics = trainer.evaluate(prep.test_data, prep.user_sequences, ds.n_items)
    print(f"\n=== Test Results (Stateful-{base_type.upper()} + {encoder_type}) ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stateful_config.yaml")
    parser.add_argument("--base", default=None, choices=["mf", "ncf", "sasrec"],
                        help="Override base_model_type from config")
    parser.add_argument(
        "--encoder", default=None,
        choices=["gru", "lstm", "mean_pool", "attention_pool", "causal_transformer", "mamba"],
        help="Override encoder_type from config",
    )
    args = parser.parse_args()
    main(args.config, args.base, args.encoder)