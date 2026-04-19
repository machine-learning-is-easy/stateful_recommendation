import torch
import torch.optim as optim
from tqdm import tqdm
from .base_trainer import BaseTrainer
from utils.metrics import evaluate_ranking


class MFTrainer(BaseTrainer):
    """Trainer for Matrix Factorization with BPR loss."""

    def __init__(self, model, config, device):
        super().__init__(model, config, device)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-5),
        )
        self.max_seq_len = config.get("max_seq_len", 50)

    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc="MF train", leave=False):
            user     = batch["user"].to(self.device)
            pos_item = batch["pos_item"].to(self.device)
            neg_item = batch["neg_item"].to(self.device)

            self.optimizer.zero_grad()
            loss = self.model(user, pos_item, neg_item)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def evaluate(self, eval_data, user_sequences, n_items):
        return evaluate_ranking(
            model=self.model,
            eval_data=eval_data,
            user_sequences=user_sequences,
            n_items=n_items,
            device=self.device,
            k_list=self.config.get("k_list", [5, 10, 20]),
            n_neg_candidates=self.config.get("n_neg_candidates", 100),
            model_type="mf",
            max_seq_len=self.max_seq_len,
        )