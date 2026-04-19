import torch
import torch.nn as nn


class UserStateEncoder(nn.Module):
    """
    Encodes a user's purchase/interaction history into a dense state vector.

    Architecture:
        item_emb  →  GRU  →  state_proj  →  user_state_emb

    The GRU is run over the (padded, sorted) sequence and the hidden
    state at the last actual time-step is used as the user state.

    This vector can replace (or augment) the static user embedding in
    any downstream recommendation model.
    """

    def __init__(
        self,
        n_items:       int,
        emb_dim:       int   = 64,
        hidden_dim:    int   = 128,
        n_layers:      int   = 2,
        dropout:       float = 0.2,
        output_dim:    int   = None,    # defaults to emb_dim if None
        bidirectional: bool  = False,
    ):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.n_layers      = n_layers
        self.bidirectional = bidirectional
        output_dim         = output_dim or emb_dim

        # n_items + 1 because idx 0 is the padding token (masked during GRU pack)
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)

        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            # inter-layer dropout only makes sense when there are ≥2 layers
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # project GRU output to the same dim as the base model's embeddings
        gru_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.state_proj = nn.Sequential(
            nn.Linear(gru_out_dim, output_dim),
            nn.LayerNorm(output_dim),   # normalise before injection to stabilise training
            nn.Tanh(),                  # bound the state to [-1, 1] like embedding init range
        )
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, std=0.02)
        for name, p in self.gru.named_parameters():
            if "weight" in name:
                # orthogonal init reduces vanishing/exploding gradients in deep GRUs
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    # ------------------------------------------------------------------
    def forward(self, seq: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq:     (B, L) padded item-id sequences (0 = pad)
            seq_len: (B,)   actual sequence lengths

        Returns:
            state_emb: (B, output_dim) — user state embeddings
        """
        x = self.dropout(self.item_emb(seq))  # (B, L, E)

        # pack_padded_sequence skips padding tokens during the GRU forward pass,
        # which is both faster and prevents padding from polluting the hidden state
        seq_len_cpu = seq_len.clamp(min=1).cpu()  # must be on CPU for pack call
        packed      = nn.utils.rnn.pack_padded_sequence(
            x, seq_len_cpu, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)  # hidden: (n_layers * dirs, B, H)

        if self.bidirectional:
            # last layer forward [-2] and backward [-1] hidden states concatenated
            h = torch.cat([hidden[-2], hidden[-1]], dim=-1)  # (B, 2H)
        else:
            # only the deepest layer's hidden state carries the full sequence context
            h = hidden[-1]  # (B, H)

        state_emb = self.state_proj(h)  # (B, output_dim)
        return state_emb