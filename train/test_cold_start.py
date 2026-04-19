"""
Smoke-test for cold-start blending — self-contained, no training dependencies.

Usage:
    python3 test_cold_start.py
    python3 test_cold_start.py --model /home/appuser/work/best_gru4rec.pt
    python3 test_cold_start.py --model /path/model.pt --pop /path/popularity.npy
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

# ── args ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=str(HERE.parent / "best_gru4rec.pt"))
parser.add_argument("--pop",   default=str(HERE / ".cache_gru4rec" / "popularity.npy"))
parser.add_argument("--top-n", type=int, default=10)
parser.add_argument("--ramp",  type=int, default=3)
args = parser.parse_args()

print(f"\n{'='*60}")
print(f"Model : {args.model}")
print(f"Pop   : {args.pop}")
print(f"Ramp  : {args.ramp}")
print(f"Top-N : {args.top_n}")
print(f"{'='*60}\n")


# ── minimal GRU4Rec (mirrors serving/_shared/model.py) ───────────────────────
class GRU4Rec(nn.Module):
    def __init__(self, num_items, cfg):
        super().__init__()
        ed, hd = cfg["embedding_dim"], cfg["hidden_dim"]
        self.item_emb    = nn.Embedding(num_items + 1, ed, padding_idx=0)
        self.emb_dropout = nn.Dropout(0.0)
        self.gru         = nn.GRU(ed, hd, num_layers=cfg["num_layers"], batch_first=True)
        self.dropout     = nn.Dropout(0.0)
        self.output_proj = nn.Linear(hd, ed, bias=False)
        self.layer_norm  = nn.LayerNorm(ed)

    def encode_session(self, prefix, user_idxs):
        x       = self.emb_dropout(self.item_emb(prefix))
        lengths = (prefix != 0).sum(dim=1).cpu().clamp(min=1)
        packed  = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, h_n  = self.gru(packed)
        return self.layer_norm(self.output_proj(self.dropout(h_n[-1])))


# ── load model ────────────────────────────────────────────────────────────────
state     = torch.load(args.model, map_location="cpu", weights_only=True)
num_items = state["item_emb.weight"].shape[0] - 1
ed        = state["item_emb.weight"].shape[1]
hd        = state["gru.weight_hh_l0"].shape[1]
nl        = sum(1 for k in state if k.startswith("gru.weight_hh_l"))

cfg   = {"embedding_dim": ed, "hidden_dim": hd, "num_layers": nl}
model = GRU4Rec(num_items, cfg)
model.load_state_dict(state, strict=True)
model.eval()
all_item_emb = model.item_emb.weight[1:].detach()

print(f"Model loaded: {num_items:,} items  embed={ed}  hidden={hd}  layers={nl}\n")


# ── load popularity ───────────────────────────────────────────────────────────
pop_available = Path(args.pop).exists()
if pop_available:
    pop_scores = torch.from_numpy(np.load(args.pop).astype("float32"))
    print(f"Popularity loaded: shape={tuple(pop_scores.shape)}  "
          f"nonzero={int((pop_scores > 0).sum()):,}\n")
else:
    print(f"[WARN] popularity.npy not found at {args.pop} — cold-start tests skipped\n")


# ── inference helpers ─────────────────────────────────────────────────────────
def alpha(session_len):
    return min(session_len / args.ramp, 1.0)


@torch.no_grad()
def predict(prefix_list, use_cold_start=True):
    prefix = torch.tensor([prefix_list], dtype=torch.long)
    users  = torch.zeros(1, dtype=torch.long)
    slen   = len(prefix_list)

    repr_     = model.encode_session(prefix, users)
    gru_scores = (repr_ @ all_item_emb.T)[0]

    if use_cold_start and pop_available:
        a      = alpha(slen)
        gru_l  = torch.log_softmax(gru_scores, dim=-1)
        pop_l  = torch.log_softmax(pop_scores, dim=-1)
        scores = a * gru_l + (1 - a) * pop_l
    else:
        a      = 1.0
        scores = gru_scores

    top = torch.topk(scores, args.top_n)
    return (
        [int(i) + 1 for i in top.indices.tolist()],
        [round(float(s), 4) for s in top.values.tolist()],
        round(a, 3),
    )


# ── tests ─────────────────────────────────────────────────────────────────────
ITEMS = [1, 2, 3, 4, 5]

cases = [
    ("Cold start  — session_len=1, alpha=0.33", ITEMS[:1]),
    ("Short       — session_len=2, alpha=0.67", ITEMS[:2]),
    ("Full ramp   — session_len=3, alpha=1.00", ITEMS[:3]),
    ("Normal      — session_len=5, alpha=1.00", ITEMS[:5]),
]

for label, prefix in cases:
    gru_idxs, gru_sc, _  = predict(prefix, use_cold_start=False)
    cs_idxs,  cs_sc,  a  = predict(prefix, use_cold_start=True)
    overlap = len(set(gru_idxs) & set(cs_idxs))
    mode    = "pure GRU" if a == 1.0 else ("pure popularity" if a == 0.0 else "blended")

    print(f"── {label} ──")
    print(f"  GRU-only  : {gru_idxs}  top-score={gru_sc[0]}")
    if pop_available:
        print(f"  Cold-start: {cs_idxs}  top-score={cs_sc[0]}")
        print(f"  alpha={a}  overlap={overlap}/{args.top_n}  [{mode}]")
    print()

# pure-popularity sanity (alpha=0)
if pop_available:
    print("── Popularity-only sanity (empty prefix → alpha=0.0) ──")
    pop_top = torch.topk(torch.log_softmax(pop_scores, dim=-1), args.top_n)
    pop_idxs = [int(i) + 1 for i in pop_top.indices.tolist()]
    print(f"  Top-{args.top_n} popular items: {pop_idxs}\n")

print("Done.")
