import torch
import torch.nn as nn
from .user_state_encoder import UserStateEncoder
from .matrix_factorization import MatrixFactorization
from .ncf import NeuralCollaborativeFiltering
from .sasrec import SASRec


class StatefulRecommender(nn.Module):
    """
    Stateful Recommendation Model.

    Pipeline:
        purchase/interaction history
              ↓
        UserStateEncoder  (pluggable encoder over item sequences)
              ↓
        user_state_emb  (B, emb_dim)
              ↓
        Base Recommender  [MF | NCF | SASRec]
              ↓
        item scores / BPR loss

    The state embedding replaces the static user embedding inside the
    chosen base model, making the recommendation *dynamic*: the same user
    gets a different representation depending on their current history.

    Supported base models:   "mf", "ncf", "sasrec"
    Supported encoder types: see UserStateEncoder.SUPPORTED
    """

    SUPPORTED = ("mf", "ncf", "sasrec")

    def __init__(
        self,
        base_model_type: str,
        n_users:         int,
        n_items:         int,
        emb_dim:         int   = 64,
        hidden_dim:      int   = 128,
        n_layers:        int   = 2,
        max_seq_len:     int   = 50,
        dropout:         float = 0.2,
        encoder_type:    str   = "gru",
        bidirectional:   bool  = False,
        d_state:         int   = 16,
        enc_n_heads:     int   = 4,
        **base_kwargs,           # forwarded to the base model constructor
    ):
        super().__init__()
        assert base_model_type in self.SUPPORTED, f"base_model must be one of {self.SUPPORTED}"
        self.base_model_type = base_model_type

        # encoder produces a (B, emb_dim) state that matches the base model's embedding dim
        self.encoder = UserStateEncoder(
            n_items=n_items,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            output_dim=emb_dim,    # must match base model emb_dim for injection
            encoder_type=encoder_type,
            bidirectional=bidirectional,
            max_seq_len=max_seq_len,
            d_state=d_state,
            enc_n_heads=enc_n_heads,
        )

        # instantiate the chosen base recommender
        if base_model_type == "mf":
            self.base_model = MatrixFactorization(
                n_users=n_users, n_items=n_items, emb_dim=emb_dim,
                dropout=dropout, **base_kwargs
            )
        elif base_model_type == "ncf":
            self.base_model = NeuralCollaborativeFiltering(
                n_users=n_users, n_items=n_items, emb_dim=emb_dim,
                dropout=dropout, **base_kwargs
            )
        elif base_model_type == "sasrec":
            # SASRec already models sequences internally; state is injected as a bias
            self.base_model = SASRec(
                n_items=n_items, emb_dim=emb_dim,
                max_seq_len=max_seq_len, dropout=dropout, **base_kwargs
            )

    # ------------------------------------------------------------------
    def forward(self, user, pos_item, neg_item, seq, seq_len):
        """
        Args:
            user:      (B,) user indices
            pos_item:  (B,) positive item indices
            neg_item:  (B,) negative item indices
            seq:       (B, L) purchase history (padded, 0=pad)
            seq_len:   (B,) actual history lengths

        Returns:
            BPR loss scalar
        """
        # encode history into a dynamic user state (B, D)
        state_emb = self.encoder(seq, seq_len)

        if self.base_model_type == "sasrec":
            # SASRec already processes the sequence; state is injected as additive context
            loss = self.base_model(
                seq=seq, pos_item=pos_item, neg_item=neg_item,
                seq_len=seq_len, user_state_emb=state_emb,
            )
        else:
            # MF and NCF: state_emb fully replaces the static user embedding
            loss = self.base_model(
                user=user, pos_item=pos_item, neg_item=neg_item,
                user_state_emb=state_emb,
            )
        return loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, user, item_ids, seq, seq_len):
        """
        Score a single user (or batch) against candidate items.

        Args:
            user:      (B,) or scalar
            item_ids:  (n_candidates,) candidate item ids
            seq:       (B, L) purchase history
            seq_len:   (B,)

        Returns:
            scores: (B, n_candidates) or (n_candidates,)
        """
        # encode current history — this reflects the user's *current* state
        state_emb = self.encoder(seq, seq_len)

        if self.base_model_type == "sasrec":
            # SASRec.predict returns (B, n_candidates) directly
            scores = self.base_model.predict(
                seq=seq, item_ids=item_ids,
                seq_len=seq_len, user_state_emb=state_emb,
            )
        else:
            # MF / NCF predict is per-user; loop over batch to collect scores
            if isinstance(user, int):
                user = torch.tensor([user], device=seq.device)

            B = state_emb.shape[0]
            scores = []
            for b in range(B):
                u_state  = state_emb[b].unsqueeze(0)
                u_tensor = user[b].unsqueeze(0) if user.dim() > 0 else user.unsqueeze(0)
                s = self.base_model.predict(u_tensor, item_ids, user_state_emb=u_state)
                scores.append(s)
            scores = torch.stack(scores, dim=0)  # (B, n_candidates)

        return scores

    # ------------------------------------------------------------------
    def get_state(self, seq: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """Expose raw user state embeddings for analysis / downstream use."""
        with torch.no_grad():
            return self.encoder(seq, seq_len)