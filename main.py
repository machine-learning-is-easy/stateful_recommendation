"""
Stateful Recommendation — unified entry point.

Usage:
    python main.py --model mf
    python main.py --model ncf
    python main.py --model sasrec
    python main.py --model stateful --base mf
    python main.py --model stateful --base ncf  --encoder lstm
    python main.py --model stateful --base sasrec --encoder mamba
    python main.py --model stateful --base mf   --encoder causal_transformer
"""

import argparse
import importlib


_MODEL_CONFIG = {
    "mf":       ("train_mf",       "configs/mf_config.yaml"),
    "ncf":      ("train_ncf",      "configs/ncf_config.yaml"),
    "sasrec":   ("train_sasrec",   "configs/sasrec_config.yaml"),
    "stateful": ("train_stateful", "configs/stateful_config.yaml"),
}


def main():
    parser = argparse.ArgumentParser(description="Stateful Recommendation System")
    parser.add_argument("--model", required=True, choices=list(_MODEL_CONFIG))
    parser.add_argument("--base", default="mf", choices=["mf", "ncf", "sasrec"],
                        help="Base model for stateful (ignored for other models)")
    parser.add_argument(
        "--encoder", default=None,
        choices=["gru", "lstm", "mean_pool", "attention_pool", "causal_transformer", "mamba"],
        help="Encoder type for stateful (ignored for other models)",
    )
    parser.add_argument("--config", default=None, help="Override config YAML path")
    args = parser.parse_args()

    module_name, default_config = _MODEL_CONFIG[args.model]
    config_path = args.config or default_config

    module = importlib.import_module(module_name)

    if args.model == "stateful":
        module.main(config_path, base_override=args.base, encoder_override=args.encoder)
    else:
        module.main(config_path)


if __name__ == "__main__":
    main()