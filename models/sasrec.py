import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PointWiseFeedForward(nn.Module):
    """Position-wise FFN used inside each SASRec transformer block."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        # expand → GELU → contract; GELU is smoother than ReLU for attention models
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SASRecBlock(nn.Module):
    """
    One transformer encoder block with pre-norm and causal self-attention.
    Pre-norm (norm before attention) stabilises training on short sequences.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff    = PointWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # --- self-attention sub-layer (pre-norm + residual) ---
        residual = x
        x = self.norm1(x)
        # attn_mask enforces causality: position i cannot attend to position j > i
        x, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.dropout(x) + residual

        # --- feed-forward sub-layer (pre-norm + residual) ---
        residual = x
        x = self.norm2(x)
        x = self.ff(x) + residual
        return x


class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation (Kang & McAuley, 2018).

    Input: padded item-id sequence of length max_seq_len.
    Output: next-item score or BPR loss.

    user_state_emb (B, D) can be injected — it is added to the positional
    encoding at position 0 (as a global context token) for stateful use.
    """

    def __init__(
        self,
        n_items:      int,
        emb_dim:      int   = 64,
        max_seq_len:  int   = 50,
        n_blocks:     int   = 2,
        n_heads:      int   = 2,
        dropout:      float = 0.2,
    ):
        super().__init__()
        self.n_items     = n_items
        self.emb_dim     = emb_dim
        self.max_seq_len = max_seq_len

        # n_items + 1 because idx 0 is the padding token
        self.item_emb    = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        # learnable positional embedding (one vector per time-step)
        self.pos_emb     = nn.Embedding(max_seq_len, emb_dim)
        self.emb_dropout = nn.Dropout(dropout)

        # stack of transformer blocks; d_ff = 4 × d_model follows the original paper
        self.blocks = nn.ModuleList([
            SASRecBlock(emb_dim, n_heads, emb_dim * 4, dropout)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(emb_dim)  # final normalisation before scoring
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,  std=0.02)

    # ------------------------------------------------------------------
    def _encode_sequence(self, seq, user_state_emb=None):
        """
        seq: (B, L) padded item ids
        Returns encoded sequence (B, L, D).
        """
        B, L = seq.shape
        # position indices 0..L-1 broadcast across the batch
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, -1)

        x = self.item_emb(seq) + self.pos_emb(positions)  # (B, L, D)

        if user_state_emb is not None:
            # broadcast state across all positions so every token is conditioned on history
            x = x + user_state_emb.unsqueeze(1)

        x = self.emb_dropout(x)

        # upper-triangular causal mask: position i cannot see j > i (autoregressive)
        causal_mask = torch.triu(
            torch.ones(L, L, device=seq.device), diagonal=1
        ).bool()

        # True positions in key_padding_mask are ignored by attention
        key_padding_mask = (seq == 0)

        for block in self.blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)

        return self.norm(x)  # (B, L, D)

    # ------------------------------------------------------------------
    def forward(self, seq, pos_item, neg_item=None, seq_len=None, user_state_emb=None):
        """
        seq:      (B, L) history
        pos_item: (B,)
        neg_item: (B,) optional, for BPR
        seq_len:  (B,) actual lengths (used to pick last valid token)
        """
        encoded = self._encode_sequence(seq, user_state_emb)  # (B, L, D)

        # index of the last real (non-padding) token per sample
        if seq_len is not None:
            idx = (seq_len - 1).clamp(min=0)
        else:
            idx = torch.tensor([seq.shape[1] - 1] * seq.shape[0], device=seq.device)

        # gather the hidden state at the last real position → user representation
        gather_idx = idx.view(-1, 1, 1).expand(-1, 1, self.emb_dim)
        user_repr  = encoded.gather(1, gather_idx).squeeze(1)  # (B, D)

        i_pos      = self.item_emb(pos_item)
        pos_scores = (user_repr * i_pos).sum(dim=-1)

        if neg_item is None:
            return pos_scores

        i_neg      = self.item_emb(neg_item)
        neg_scores = (user_repr * i_neg).sum(dim=-1)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        return loss

    def predict(self, seq, item_ids, seq_len=None, user_state_emb=None):
        """Score last-position repr against candidate items."""
        encoded = self._encode_sequence(seq, user_state_emb)

        if seq_len is not None:
            idx = (seq_len - 1).clamp(min=0)
        else:
            idx = torch.tensor([seq.shape[1] - 1] * seq.shape[0], device=seq.device)

        gather_idx = idx.view(-1, 1, 1).expand(-1, 1, self.emb_dim)
        user_repr  = encoded.gather(1, gather_idx).squeeze(1)    # (B, D)

        items  = self.item_emb(item_ids)                         # (n_candidates, D)
        # batched dot product: each user row vs every candidate column
        return torch.matmul(user_repr, items.T)                  # (B, n_candidates)
