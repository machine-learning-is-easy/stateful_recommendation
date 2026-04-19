import torch
import torch.optim as optim
from tqdm import tqdm
from .base_trainer import BaseTrainer
from utils.metrics import evaluate_ranking


class StatefulTrainer(BaseTrainer):
    """
    Trainer for StatefulRecommender.

    The stateful model jointly trains the UserStateEncoder (GRU) and the
    chosen base recommender end-to-end with BPR loss.
    """

    def __init__(self, model, config, device):
        super().__init__(model, config, device)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get("lr", 5e-4),
            weight_decay=config.get("weight_decay", 1e-5),
        )
        warmup = config.get("warmup_steps", 200)
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / max(warmup, 1)),
        )
        self.max_seq_len = config.get("max_seq_len", 50)
        self._step = 0

    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc="Stateful train", leave=False):
            user     = batch["user"].to(self.device)
            pos_item = batch["pos_item"].to(self.device)
            neg_item = batch["neg_item"].to(self.device)
            seq      = batch["seq"].to(self.device)
            seq_len  = batch["seq_len"].to(self.device)

            self.optimizer.zero_grad()
            loss = self.model(user, pos_item, neg_item, seq, seq_len)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.scheduler.step()
            self._step += 1
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
            model_type="stateful",
            max_seq_len=self.max_seq_len,
        )