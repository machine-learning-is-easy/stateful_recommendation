import torch
import torch.nn as nn


class MatrixFactorization(nn.Module):
    """
    Bayesian Personalized Ranking Matrix Factorization (BPR-MF).

    Learns latent user and item embeddings optimized with BPR loss.
    Accepts an optional precomputed user_state_emb to override the
    static user embedding — used by StatefulRecommender.
    """

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        # separate embedding tables for users and items; idx 0 reserved for padding
        self.user_emb = nn.Embedding(n_users, emb_dim, padding_idx=0)
        self.item_emb = nn.Embedding(n_items, emb_dim, padding_idx=0)
        self.dropout  = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        # small normal init keeps dot products well-scaled at the start of training
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    # ------------------------------------------------------------------
    def forward(self, user, pos_item, neg_item=None, user_state_emb=None):
        """
        Args:
            user:           (B,) user indices
            pos_item:       (B,) positive item indices
            neg_item:       (B,) negative item indices (optional, for BPR)
            user_state_emb: (B, emb_dim) external user state (overrides user_emb)

        Returns:
            If neg_item provided: BPR loss scalar.
            Otherwise:           pos scores (B,).
        """
        # prefer injected state embedding over the stored static user embedding
        u     = user_state_emb if user_state_emb is not None else self.dropout(self.user_emb(user))
        i_pos = self.dropout(self.item_emb(pos_item))

        # dot product similarity between user and positive item
        pos_scores = (u * i_pos).sum(dim=-1)

        if neg_item is None:
            # inference path — return raw scores for ranking
            return pos_scores

        i_neg      = self.dropout(self.item_emb(neg_item))
        neg_scores = (u * i_neg).sum(dim=-1)

        # BPR loss: maximise score gap between positive and negative item
        # eps added inside log for numerical stability
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        return loss

    def predict(self, user, item_ids, user_state_emb=None):
        """Score a user against a set of candidate items."""
        u = user_state_emb if user_state_emb is not None else self.user_emb(user)
        if u.dim() == 1:
            # ensure user vector is (1, D) for broadcasting against item matrix
            u = u.unsqueeze(0)
        items = self.item_emb(item_ids)     # (n_candidates, D)
        return (u * items).sum(dim=-1)      # (n_candidates,)