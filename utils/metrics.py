import math
import numpy as np
import torch
from typing import List, Dict


def hit_rate_at_k(ranked_list: List[int], ground_truth: int, k: int) -> float:
    return 1.0 if ground_truth in ranked_list[:k] else 0.0


def ndcg_at_k(ranked_list: List[int], ground_truth: int, k: int) -> float:
    if ground_truth in ranked_list[:k]:
        rank = ranked_list.index(ground_truth)
        return 1.0 / math.log2(rank + 2)
    return 0.0


def evaluate_ranking(
    model,
    eval_data: List[tuple],
    user_sequences: Dict[int, List[int]],
    n_items: int,
    device: torch.device,
    k_list: List[int] = None,
    n_neg_candidates: int = 100,
    model_type: str = "mf",
    max_seq_len: int = 50,
) -> Dict[str, float]:
    """
    Evaluates a recommendation model using sampled negative evaluation.

    For each (user, target_item, history) triple:
      - Randomly sample n_neg_candidates negative items
      - Rank target_item among negatives + target
      - Compute HR@K and NDCG@K
    """
    if k_list is None:
        k_list = [5, 10, 20]

    model.eval()
    metrics = {f"HR@{k}": [] for k in k_list}
    metrics.update({f"NDCG@{k}": [] for k in k_list})

    rng = np.random.default_rng(42)

    with torch.no_grad():
        for user, target_item, history in eval_data:
            user_items = set(user_sequences.get(user, []))
            # sample negatives
            neg_pool = list(set(range(1, n_items)) - user_items - {target_item})
            if len(neg_pool) < n_neg_candidates:
                continue
            negs = rng.choice(neg_pool, n_neg_candidates, replace=False).tolist()
            candidates = [target_item] + negs  # target always at index 0

            seq_pad = history[-max_seq_len:]
            seq_len = len(seq_pad)
            seq_pad = [0] * (max_seq_len - seq_len) + seq_pad

            seq_t   = torch.tensor([seq_pad], dtype=torch.long, device=device)
            seq_len_t = torch.tensor([seq_len], dtype=torch.long, device=device)
            user_t  = torch.tensor([user], dtype=torch.long, device=device)
            cand_t  = torch.tensor(candidates, dtype=torch.long, device=device)

            if hasattr(model, "get_state"):  # StatefulRecommender
                scores = model.predict(user_t, cand_t, seq_t, seq_len_t)
                scores = scores.squeeze(0)
            elif model_type == "sasrec":
                scores = model.predict(seq_t, cand_t, seq_len=seq_len_t)
                scores = scores.squeeze(0)
            else:
                u_emb = model.user_emb(user_t) if hasattr(model, "user_emb") else None
                scores = model.predict(user_t, cand_t, user_state_emb=u_emb)

            ranked_indices = scores.argsort(descending=True).tolist()
            ranked_items   = [candidates[i] for i in ranked_indices]

            for k in k_list:
                metrics[f"HR@{k}"].append(hit_rate_at_k(ranked_items, target_item, k))
                metrics[f"NDCG@{k}"].append(ndcg_at_k(ranked_items, target_item, k))

    return {key: float(np.mean(vals)) for key, vals in metrics.items() if vals}