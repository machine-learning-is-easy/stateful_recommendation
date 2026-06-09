import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List
from torch.utils.tensorboard import SummaryWriter


class BaseTrainer(ABC):
    def __init__(self, model: nn.Module, config: dict, device: torch.device):
        self.model  = model.to(device)
        self.config = config
        self.device = device
        self.writer = SummaryWriter(log_dir=config.get("log_dir", "runs/exp"))
        self.best_metric = 0.0
        self.best_epoch  = 0

    # ------------------------------------------------------------------
    @abstractmethod
    def train_epoch(self, loader) -> float:
        ...

    @abstractmethod
    def evaluate(self, eval_data, user_sequences, n_items) -> Dict[str, float]:
        ...

    # ------------------------------------------------------------------
    def fit(self, train_loader, val_data, user_sequences, n_items):
        n_epochs  = self.config.get("n_epochs", 50)
        patience  = self.config.get("patience", 10)
        monitor   = self.config.get("monitor", "HR@10")
        no_improve = 0

        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_data, user_sequences, n_items)

            self.writer.add_scalar("Loss/train", train_loss, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"Val/{k}", v, epoch)

            score = val_metrics.get(monitor, 0.0)
            if score > self.best_metric:
                self.best_metric = score
                self.best_epoch  = epoch
                self._save_checkpoint()
                no_improve = 0
            else:
                no_improve += 1

            print(
                f"Epoch {epoch:3d} | loss={train_loss:.4f} | "
                + " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                + f" | best {monitor}={self.best_metric:.4f} (ep {self.best_epoch})"
            )

            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        self.writer.close()
        return self.best_metric

    def _save_checkpoint(self):
        path = self.config.get("checkpoint_path", "checkpoints/best.pt")
        import os; os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_best(self):
        path = self.config.get("checkpoint_path", "checkpoints/best.pt")
        if not os.path.exists(path):
            print(f"[BaseTrainer] No checkpoint at {path}; using current weights.")
            return
        self.model.load_state_dict(torch.load(path, map_location=self.device))