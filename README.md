# Stateful Recommendation System

A PyTorch implementation of stateful sequential recommendation using the Amazon review dataset. Three base recommendation models are provided, each enhanced by a GRU-based **UserStateEncoder** that replaces static user embeddings with dynamic state embeddings derived from purchase history.

---

## Architecture Overview

### Base Models

| Model | Type | Key Idea |
|-------|------|----------|
| **MF** (BPR-MF) | Collaborative Filtering | Dot-product of user/item embeddings, optimized with BPR loss |
| **NCF** (NeuralMF) | Neural CF | Combines GMF (element-wise product) and MLP tower, fused at output |
| **SASRec** | Sequential | Causal self-attention over item history, predicts next item |

### Stateful Recommendation Model

```
Purchase / Interaction History
  [item₁, item₂, ..., itemₙ]
          │
          ▼
  ┌─────────────────────┐
  │   UserStateEncoder  │  ItemEmbedding → GRU (n_layers) → Linear + LayerNorm
  └─────────────────────┘
          │
          ▼  user_state_emb  (B, emb_dim)
          │
          ▼
  ┌─────────────────────┐
  │    Base Recommender │  MF  │  NCF  │  SASRec
  │  (state_emb replaces│
  │   static user emb)  │
  └─────────────────────┘
          │
          ▼
     Item Scores / BPR Loss
```

The `UserStateEncoder` encodes the user's chronological interaction history into a single dense vector. This vector is injected into the base model in place of (or in addition to) the static user embedding, making recommendations **dynamic** — the same user gets a different representation as their history evolves.

---

## Project Structure

```
stateful_recommendation/
├── data/
│   ├── amazon_dataset.py     # Downloads, caches, and encodes Amazon review data
│   └── preprocessor.py       # Leave-one-out split, SequenceDataset, PairwiseDataset
├── models/
│   ├── matrix_factorization.py   # BPR-MF
│   ├── ncf.py                    # NeuralMF (GMF + MLP)
│   ├── sasrec.py                 # Self-Attentive Sequential Recommender
│   ├── user_state_encoder.py     # GRU-based history encoder
│   └── stateful_rec.py           # StatefulRecommender (encoder + base model)
├── trainers/
│   ├── base_trainer.py           # Abstract trainer with early stopping + TensorBoard
│   ├── mf_trainer.py
│   ├── ncf_trainer.py
│   ├── sasrec_trainer.py
│   └── stateful_trainer.py
├── utils/
│   ├── metrics.py                # HR@K, NDCG@K, sampled ranking evaluation
│   └── data_utils.py
├── configs/
│   ├── mf_config.yaml
│   ├── ncf_config.yaml
│   ├── sasrec_config.yaml
│   └── stateful_config.yaml
├── train_mf.py
├── train_ncf.py
├── train_sasrec.py
├── train_stateful.py
├── main.py
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch 2.0+, CUDA optional.

---

## Dataset

Amazon product review datasets are downloaded automatically from the UCSD McAuley Lab on first run and cached as Parquet files under `data/raw/`.

Supported categories (set `category` in any config YAML):

| Key | Dataset |
|-----|---------|
| `beauty` | All Beauty (default, ~200K reviews) |
| `sports` | Sports and Outdoors |
| `toys` | Toys and Games |
| `movies` | Movies and TV |
| `ml-1m` | Synthetic fallback (no download needed) |

All datasets are k-core filtered (default `min_interactions: 5`) and split using **leave-one-out**: last item → test, second-to-last → val, rest → train.

---

## Usage

### Unified entry point

```bash
# Base models
python main.py --model mf
python main.py --model ncf
python main.py --model sasrec

# Stateful model with different base recommenders
python main.py --model stateful --base mf
python main.py --model stateful --base ncf
python main.py --model stateful --base sasrec

# Custom config
python main.py --model stateful --base ncf --config configs/stateful_config.yaml
```

### Individual training scripts

```bash
python train_mf.py      --config configs/mf_config.yaml
python train_ncf.py     --config configs/ncf_config.yaml
python train_sasrec.py  --config configs/sasrec_config.yaml
python train_stateful.py --config configs/stateful_config.yaml --base mf
```

### TensorBoard

```bash
tensorboard --logdir runs/
```

---

## Configuration

All hyperparameters are controlled via YAML files in `configs/`. Example (`stateful_config.yaml`):

```yaml
dataset:
  category: beauty
  max_seq_len: 50
  min_interactions: 5

model:
  base_model_type: mf     # mf | ncf | sasrec
  emb_dim: 64
  hidden_dim: 128          # GRU hidden size
  gru_layers: 2
  dropout: 0.2

training:
  batch_size: 256
  lr: 0.0005
  n_epochs: 50
  patience: 10
  monitor: HR@10
```

---

## Evaluation

Models are evaluated using **sampled ranking** (100 random negatives per test user):

- **HR@K** — Hit Rate: fraction of users where the target item appears in the top-K
- **NDCG@K** — Normalized Discounted Cumulative Gain: rank-aware measure

Checkpoints are saved to `checkpoints/` whenever validation `HR@10` improves.

---

## Model Details

### UserStateEncoder (`models/user_state_encoder.py`)

| Component | Detail |
|-----------|--------|
| Input | Padded item-id sequence `(B, L)` |
| Item embedding | `nn.Embedding(n_items, emb_dim)` |
| Recurrent layer | GRU, `n_layers=2`, `hidden_dim=128` |
| Output projection | `Linear → LayerNorm → Tanh` |
| Output | User state vector `(B, emb_dim)` |

### StatefulRecommender (`models/stateful_rec.py`)

The encoder and base model are trained **end-to-end** with BPR loss. At inference, `predict()` computes state embeddings on-the-fly from the user's current history and scores all candidate items.

```python
from models.stateful_rec import StatefulRecommender

model = StatefulRecommender(
    base_model_type="ncf",
    n_users=n_users,
    n_items=n_items,
    emb_dim=64,
    hidden_dim=128,
    gru_layers=2,
)

# Forward (training)
loss = model(user, pos_item, neg_item, seq, seq_len)

# Inference
scores = model.predict(user, candidate_items, seq, seq_len)  # (B, n_candidates)

# Inspect raw user state
state = model.get_state(seq, seq_len)  # (B, emb_dim)
```

---

## References

- He et al. (2017) — [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)
- Kang & McAuley (2018) — [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)
- Rendle et al. (2009) — [BPR: Bayesian Personalized Ranking](https://arxiv.org/abs/1205.2618)
- McAuley et al. — [Amazon Review Dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)