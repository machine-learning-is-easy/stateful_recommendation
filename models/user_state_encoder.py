"""
User State Encoders.

All encoders share the same public interface:
    UserStateEncoder.forward(seq, seq_len) -> (B, output_dim)

Supported encoder_type values
    "gru"                – Gated Recurrent Unit (original baseline)
    "lstm"               – Long Short-Term Memory
    "mean_pool"          – Parameter-free masked mean pooling (ablation floor)
    "attention_pool"     – Attention-weighted pooling with a learnable query
    "causal_transformer" – Left-to-right transformer; last token = user state
    "mamba"              – Selective State Space Model (Mamba-style, pure PyTorch)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── encoder cores ─────────────────────────────────────────────────────────────
# All cores share the signature:
#   forward(x, seq, seq_len) -> (B, out_dim)
#
#   x       – (B, L, emb_dim)  embedded + dropped-out item sequences
#   seq     – (B, L)           original token ids  (0 = pad)
#   seq_len – (B,)             actual sequence lengths


class _GRUCore(nn.Module):
    def __init__(self, emb_dim, hidden_dim, n_layers, dropout, bidirectional):
        super().__init__()
        self.bidirectional = bidirectional
        self.out_dim = hidden_dim * (2 if bidirectional else 1)
        self.gru = nn.GRU(
            emb_dim, hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        for name, p in self.gru.named_parameters():
            nn.init.orthogonal_(p) if "weight" in name else nn.init.zeros_(p)

    def forward(self, x, seq, seq_len):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, seq_len.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)  # (n_layers * dirs, B, H)
        return torch.cat([h[-2], h[-1]], dim=-1) if self.bidirectional else h[-1]


class _LSTMCore(nn.Module):
    def __init__(self, emb_dim, hidden_dim, n_layers, dropout, bidirectional):
        super().__init__()
        self.bidirectional = bidirectional
        self.out_dim = hidden_dim * (2 if bidirectional else 1)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        for name, p in self.lstm.named_parameters():
            nn.init.orthogonal_(p) if "weight" in name else nn.init.zeros_(p)

    def forward(self, x, seq, seq_len):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, seq_len.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)  # h: (n_layers * dirs, B, H)
        return torch.cat([h[-2], h[-1]], dim=-1) if self.bidirectional else h[-1]


class _MeanPoolCore(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.out_dim = emb_dim

    def forward(self, x, seq, seq_len):
        mask = (seq != 0).float().unsqueeze(-1)          # (B, L, 1)
        return (x * mask).sum(1) / mask.sum(1).clamp(min=1)  # (B, E)


class _AttentionPoolCore(nn.Module):
    """Scaled dot-product between each item embedding and a global learnable query."""

    def __init__(self, emb_dim):
        super().__init__()
        self.out_dim = emb_dim
        self.query = nn.Parameter(torch.empty(emb_dim))
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x, seq, seq_len):
        scale = x.size(-1) ** 0.5
        attn = (x * self.query).sum(-1) / scale          # (B, L)
        attn = attn.masked_fill(seq == 0, float("-inf"))
        weights = F.softmax(attn, dim=-1)                 # (B, L)
        weights = torch.nan_to_num(weights)               # all-pad edge case → zeros
        return (weights.unsqueeze(-1) * x).sum(1)         # (B, E)


class _CausalTransformerCore(nn.Module):
    """Left-to-right transformer encoder; output = representation at last actual token."""

    def __init__(self, emb_dim, n_heads, n_layers, dropout, max_seq_len):
        super().__init__()
        assert emb_dim % n_heads == 0, f"emb_dim ({emb_dim}) must be divisible by n_heads ({n_heads})"
        self.out_dim = emb_dim
        self.register_buffer("pe", self._sinusoidal_pe(max_seq_len, emb_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=emb_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _sinusoidal_pe(max_len, d_model):
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].size(1)])
        return pe  # (max_len, E)

    def forward(self, x, seq, seq_len):
        B, L, _ = x.shape
        x = self.drop(x + self.pe[:L])
        causal_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        pad_mask    = (seq == 0)  # (B, L); True where padding
        out = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)  # (B, L, E)
        idx = (seq_len - 1).clamp(min=0).long()
        return out[torch.arange(B, device=x.device), idx]  # (B, E)


# ── Mamba (Selective State Space Model) ──────────────────────────────────────

def _hillis_steele_scan(log_a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Parallel inclusive prefix scan for the linear recurrence h_t = a_t*h_{t-1} + b_t (h_-1=0).

    Uses the Hillis-Steele algorithm with the associative composition operator:
        (log_a_R, b_R) ∘ (log_a_L, b_L)  =  (log_a_L + log_a_R,  exp(log_a_R)*b_L + b_R)

    Complexity: O(L log L) work, ceil(log2 L) serial passes — vs. O(L) serial passes before.
    Numerically stable: all exp() calls are on log(a_t) ≤ 0 (a_t ∈ (0,1)), so no overflow.

    Args:
        log_a: (N, L)  log of state-transition coefficients (≤ 0 for decaying SSM)
        b:     (N, L)  input (free) terms
    Returns:
        h:     (N, L)  hidden states at each position
    """
    L = log_a.shape[-1]
    la = log_a   # accumulated log(a) for the window ending at each position
    h  = b       # accumulated free term (h when starting from 0)

    stride = 1
    while stride < L:
        # For every position t ≥ stride, compose window[0..t-stride] before window[t-stride+1..t]:
        #   new_h[t]  = exp(la[t]) * h[t-stride]  +  h[t]
        #   new_la[t] = la[t-stride] + la[t]
        la_right = la[..., stride:]           # (N, L-stride)
        h_left   = h[..., :L - stride]
        new_h    = torch.exp(la_right) * h_left + h[..., stride:]
        new_la   = la[..., :L - stride] + la_right

        # Positions [0:stride] are unchanged; append updated [stride:] part
        h  = torch.cat([h[..., :stride],  new_h],  dim=-1)
        la = torch.cat([la[..., :stride], new_la], dim=-1)

        stride *= 2

    return h  # (N, L)


class _MambaSSMLayer(nn.Module):
    """
    One Mamba block: selective SSM with input-dependent state transitions.

    Key ideas vs. GRU:
      - State transition matrix A is NOT input-dependent (stable global structure)
      - Step size Δ, input-to-state B, and output-from-state C are input-dependent
      - A local depthwise conv provides short-range context before the SSM
      - Gating (SiLU) controls information flow

    The SSM recurrence h_t = Ā_t h_{t-1} + B̄_t u_t is computed via the
    Hillis-Steele parallel scan (ceil(log2 L) serial passes of vectorised ops)
    rather than a Python for-loop over L.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner = d_inner
        self.d_state = d_state

        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(
            d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner, bias=True
        )
        self.x_proj   = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj  = nn.Linear(1, d_inner, bias=True)

        # A: (d_inner, d_state) — log-scale, negative eigenvalues ensure stability
        A_init = torch.arange(1, d_state + 1, dtype=torch.float).repeat(d_inner, 1)
        self.A_log    = nn.Parameter(torch.log(A_init))
        self.D        = nn.Parameter(torch.ones(d_inner))  # skip connection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.drop     = nn.Dropout(dropout)

    def forward(self, x):
        B, L, _ = x.shape

        xz           = self.in_proj(x)               # (B, L, 2*d_inner)
        x_ssm, z     = xz.chunk(2, dim=-1)           # each (B, L, d_inner)

        # Depthwise causal conv along sequence axis
        x_conv = self.conv1d(x_ssm.transpose(1, 2))  # (B, d_inner, L+pad)
        x_act  = F.silu(x_conv[:, :, :L].transpose(1, 2))  # (B, L, d_inner)

        # Selective (input-dependent) parameters
        bcd                    = self.x_proj(x_act)   # (B, L, 2*d_state+1)
        ssm_b, ssm_c, log_dt   = bcd.split([self.d_state, self.d_state, 1], dim=-1)
        delta                  = F.softplus(self.dt_proj(log_dt))  # (B, L, d_inner)

        # Discretise via zero-order hold
        A     = -torch.exp(self.A_log.float())            # (d_inner, d_state), negative
        log_a = delta.unsqueeze(-1) * A[None, None]       # (B, L, d_inner, d_state) = log(Ā)
        b_in  = delta.unsqueeze(-1) * ssm_b.unsqueeze(2) \
                * x_act.unsqueeze(-1)                     # (B, L, d_inner, d_state) = B̄ u_t

        if log_a.is_cuda:
            # GPU: HS parallel scan — flatten to (N, L), run ceil(log2 L) vectorised passes
            N          = B * self.d_inner * self.d_state
            log_a_flat = log_a.permute(0, 2, 3, 1).reshape(N, L)
            b_flat     = b_in.permute(0, 2, 3, 1).reshape(N, L)
            h_flat     = _hillis_steele_scan(log_a_flat, b_flat)            # (N, L)
            h_all      = h_flat.reshape(B, self.d_inner, self.d_state, L).permute(0, 3, 1, 2)
            y          = (h_all * ssm_c.unsqueeze(2)).sum(-1)               # (B, L, d_inner)
        else:
            # CPU: sequential scan keeping h small at (B, d_inner, d_state)
            # — avoids the 50× memory blowup of flattening to (N, L)
            h  = x.new_zeros(B, self.d_inner, self.d_state)
            ys = []
            for t in range(L):
                h = torch.exp(log_a[:, t]) * h + b_in[:, t]        # (B, d_inner, d_state)
                ys.append((h * ssm_c[:, t, None, :]).sum(-1))       # (B, d_inner)
            y = torch.stack(ys, dim=1)                               # (B, L, d_inner)

        # Skip connection + gate
        out = (y + self.D * x_act) * F.silu(z)
        return self.drop(self.out_proj(out))               # (B, L, d_model)


class _MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm  = _MambaSSMLayer(d_model, d_state=d_state, dropout=dropout)

    def forward(self, x):
        return x + self.ssm(self.norm(x))  # pre-norm residual


class _MambaCore(nn.Module):
    """Stack of Mamba blocks; output = last actual-token representation."""

    def __init__(self, emb_dim, d_state, n_layers, dropout):
        super().__init__()
        self.out_dim = emb_dim
        self.blocks  = nn.ModuleList([
            _MambaBlock(emb_dim, d_state, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x, seq, seq_len):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        B   = x.size(0)
        idx = (seq_len - 1).clamp(min=0).long()
        return x[torch.arange(B, device=x.device), idx]  # (B, E)


# ── public API ────────────────────────────────────────────────────────────────

class UserStateEncoder(nn.Module):
    """
    Encodes a user's interaction history into a dense state vector.

    The encoder type is selected via the `encoder_type` argument:
        "gru"                – GRU (strong sequential baseline)
        "lstm"               – LSTM (separate cell / hidden state)
        "mean_pool"          – Parameter-free mean (ablation floor)
        "attention_pool"     – Learned query attention over item embeddings
        "causal_transformer" – Left-to-right transformer; last-token output
        "mamba"              – Selective SSM with Hillis-Steele parallel scan

    All variants produce a (B, output_dim) vector compatible with any base
    recommendation model in this repo.
    """

    SUPPORTED = ("gru", "lstm", "mean_pool", "attention_pool", "causal_transformer", "mamba")

    def __init__(
        self,
        n_items:       int,
        emb_dim:       int   = 64,
        hidden_dim:    int   = 128,
        n_layers:      int   = 2,
        dropout:       float = 0.2,
        output_dim:    int   = None,
        encoder_type:  str   = "gru",
        bidirectional: bool  = False,
        max_seq_len:   int   = 50,
        d_state:       int   = 16,
        enc_n_heads:   int   = 4,
    ):
        super().__init__()
        assert encoder_type in self.SUPPORTED, \
            f"encoder_type must be one of {self.SUPPORTED}, got '{encoder_type}'"
        output_dim = output_dim or emb_dim

        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        self.dropout = nn.Dropout(dropout)

        if encoder_type == "gru":
            self._core = _GRUCore(emb_dim, hidden_dim, n_layers, dropout, bidirectional)
        elif encoder_type == "lstm":
            self._core = _LSTMCore(emb_dim, hidden_dim, n_layers, dropout, bidirectional)
        elif encoder_type == "mean_pool":
            self._core = _MeanPoolCore(emb_dim)
        elif encoder_type == "attention_pool":
            self._core = _AttentionPoolCore(emb_dim)
        elif encoder_type == "causal_transformer":
            self._core = _CausalTransformerCore(emb_dim, enc_n_heads, n_layers, dropout, max_seq_len)
        elif encoder_type == "mamba":
            self._core = _MambaCore(emb_dim, d_state, n_layers, dropout)

        self.state_proj = nn.Sequential(
            nn.Linear(self._core.out_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh(),
        )

    def forward(self, seq: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq:     (B, L)  padded item-id sequences  (0 = pad)
            seq_len: (B,)    actual sequence lengths

        Returns:
            state_emb: (B, output_dim)
        """
        x = self.dropout(self.item_emb(seq))  # (B, L, E)
        h = self._core(x, seq, seq_len)       # (B, core.out_dim)
        return self.state_proj(h)             # (B, output_dim)
