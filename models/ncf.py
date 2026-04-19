import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class NeuralCollaborativeFiltering(nn.Module):
    """
    NeuralMF: combines Generalized Matrix Factorization (GMF) and
    Multi-Layer Perceptron (MLP) as proposed by He et al. (2017).

    Accepts an optional user_state_emb (B, emb_dim) to replace the static
    user embedding — used by StatefulRecommender.
    """

    def __init__(
        self,
        n_users:     int,
        n_items:     int,
        emb_dim:     int = 64,
        mlp_layers:  List[int] = None,
        dropout:     float = 0.2,
    ):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [256, 128, 64]

        # GMF path: element-wise product of user and item embeddings
        self.gmf_user_emb = nn.Embedding(n_users, emb_dim, padding_idx=0)
        self.gmf_item_emb = nn.Embedding(n_items, emb_dim, padding_idx=0)

        # MLP path: separate embeddings concatenated before the MLP tower
        # using separate tables lets each path learn different facets of preference
        self.mlp_user_emb = nn.Embedding(n_users, emb_dim, padding_idx=0)
        self.mlp_item_emb = nn.Embedding(n_items, emb_dim, padding_idx=0)

        # MLP tower: progressively reduces concat(user, item) to a dense vector
        mlp_input_dim = emb_dim * 2
        layers = []
        for out_dim in mlp_layers:
            layers += [nn.Linear(mlp_input_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            mlp_input_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        # final fusion: concat GMF output (D) + MLP last layer (mlp_layers[-1]) → scalar
        self.output = nn.Linear(emb_dim + mlp_layers[-1], 1)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for emb in [self.gmf_user_emb, self.gmf_item_emb, self.mlp_user_emb, self.mlp_item_emb]:
            nn.init.normal_(emb.weight, std=0.01)
        # xavier keeps output layer gradients well-conditioned
        nn.init.xavier_uniform_(self.output.weight)

    # ------------------------------------------------------------------
    def _user_repr(self, user, user_state_emb):
        """Return (gmf_u, mlp_u) — replace both with state_emb when provided."""
        if user_state_emb is not None:
            # stateful mode: a single state vector feeds both GMF and MLP paths
            return user_state_emb, user_state_emb
        return self.dropout(self.gmf_user_emb(user)), self.dropout(self.mlp_user_emb(user))

    def _score(self, gmf_u, mlp_u, item):
        """Compute NeuralMF logit for a (user_repr, item) pair."""
        gmf_i = self.dropout(self.gmf_item_emb(item))
        mlp_i = self.dropout(self.mlp_item_emb(item))

        gmf_out = gmf_u * gmf_i                                    # (B, D) — linear signal
        mlp_out = self.mlp(torch.cat([mlp_u, mlp_i], dim=-1))      # (B, L) — non-linear signal
        logit   = self.output(torch.cat([gmf_out, mlp_out], dim=-1)).squeeze(-1)
        return logit

    # ------------------------------------------------------------------
    def forward(self, user, pos_item, neg_item=None, user_state_emb=None):
        gmf_u, mlp_u = self._user_repr(user, user_state_emb)

        pos_scores = self._score(gmf_u, mlp_u, pos_item)

        if neg_item is None:
            # inference path — caller handles ranking
            return pos_scores

        neg_scores = self._score(gmf_u, mlp_u, neg_item)
        # BPR loss encourages pos_score > neg_score
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        return loss

    def predict(self, user, item_ids, user_state_emb=None):
        gmf_u, mlp_u = self._user_repr(user, user_state_emb)
        if gmf_u.dim() == 1:
            # lift to (1, D) so _score broadcasting works correctly
            gmf_u = gmf_u.unsqueeze(0)
            mlp_u = mlp_u.unsqueeze(0)

        # score each candidate item individually (avoids a batched embedding gather)
        scores = []
        for item in item_ids.unsqueeze(-1) if item_ids.dim() == 1 else item_ids:
            item = item.unsqueeze(0) if item.dim() == 0 else item
            scores.append(self._score(gmf_u, mlp_u, item))
        return torch.cat(scores, dim=0)