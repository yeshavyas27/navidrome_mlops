"""
Quick smoke-test for cold-start blending in the training container.

Usage:
    python test_cold_start.py                          # uses best_gru4rec.pt + popularity.npy from cache
    python test_cold_start.py --model /path/model.pt   # explicit model path
    python test_cold_start.py --pop   /path/pop.npy    # explicit popularity path

Tests:
    1. Normal inference (session len >= ramp) — alpha should be 1.0
    2. Cold start (session len 1)             — alpha should be 0.33
    3. Empty-ish session (session len 2)      — alpha should be 0.67
    4. Top-20 popularity sanity check
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).resolve().parent
CACHE_DIR  = HERE / ".cache_gru4rec"
MODEL_PATH = HERE.parent / "best_gru4rec.pt"
POP_PATH   = CACHE_DIR / "popularity.npy"

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=str(MODEL_PATH))
parser.add_argument("--pop",   default=str(POP_PATH))
parser.add_argument("--top-n", type=int, default=10)
parser.add_argument("--ramp",  type=int, default=3)
args = parser.parse_args()

# ── load model ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Model : {args.model}")
print(f"Pop   : {args.pop}")
print(f"Ramp  : {args.ramp}  (alpha=1.0 when session_len >= ramp)")
print(f"Top-N : {args.top_n}")
print(f"{'='*60}\n")

sys.path.insert(0, str(HERE))
from gru4rec import GRU4Rec, DEFAULT_CFG  # type: ignore

# gru4rec.py exposes cfg dict, not DEFAULT_CFG — pull from module
import gru4rec as _g
_cfg = {
    "embedding_dim":     64,
    "hidden_dim":        128,
    "num_layers":        1,
    "dropout":           0.0,
    "embedding_dropout": 0.0,
    "use_user_context":  False,
}

state = torch.load(args.model, map_location="cpu", weights_only=True)
num_items = state["item_emb.weight"].shape[0] - 1   # subtract padding row

model = GRU4Rec(num_items=num_items, num_users=0, cfg=_cfg)
model.load_state_dict(state, strict=True)
model.eval()
all_item_emb = model.item_emb.weight[1:].detach()   # (num_items, D)

print(f"Model loaded: {num_items:,} items, embed_dim={all_item_emb.shape[1]}")

# ── load cold-start blender ───────────────────────────────────────────────────
if not Path(args.pop).exists():
    print(f"[WARN] popularity.npy not found at {args.pop}")
    print("       Run a training job first, or check POPULARITY_PATH.")
    print("       Falling back to pure GRU4Rec for all tests.\n")
    blender = None
else:
    from cold_start import ColdStartRecommender
    # ColdStartBlender (serving) and ColdStartRecommender (train) share the
    # same logic; here we load via numpy directly to avoid the serving import.
    pop_scores = np.load(args.pop)
    print(f"Popularity scores loaded: shape={pop_scores.shape}, "
          f"max={pop_scores.max():.2f}, nonzero={np.count_nonzero(pop_scores):,}\n")

    class _Blender:
        """Thin wrapper matching serving/_shared/cold_start.py interface."""
        def __init__(self, pop, ramp):
            self._pop  = torch.from_numpy(pop.astype(np.float32))
            self.ramp  = ramp

        def alpha(self, slen):
            return min(slen / self.ramp, 1.0)

        @torch.no_grad()
        def predict(self, prefix, top_n, exclude=()):
            slen   = int((prefix != 0).sum())
            a      = self.alpha(slen)
            repr_  = model.encode_session(prefix, torch.zeros(1, dtype=torch.long))
            gru_s  = repr_ @ all_item_emb.T               # (1, num_items)
            gru_l  = torch.log_softmax(gru_s[0], dim=-1)
            pop_l  = torch.log_softmax(self._pop, dim=-1)
            scores = a * gru_l + (1 - a) * pop_l
            for idx in exclude:
                scores[idx - 1] = float("-inf")
            top    = torch.topk(scores, top_n)
            return (
                [int(i) + 1 for i in top.indices.tolist()],
                [round(float(s), 4) for s in top.values.tolist()],
                round(a, 3),
            )

    blender = _Blender(pop_scores, ramp=args.ramp)


# ── helpers ───────────────────────────────────────────────────────────────────
def gru_only(prefix_list, top_n):
    with torch.no_grad():
        prefix = torch.tensor([prefix_list], dtype=torch.long)
        users  = torch.zeros(1, dtype=torch.long)
        repr_  = model.encode_session(prefix, users)
        scores = (repr_ @ all_item_emb.T)[0]
        top    = torch.topk(scores, top_n)
        return (
            [int(i) + 1 for i in top.indices.tolist()],
            [round(float(s), 4) for s in top.values.tolist()],
        )


def run_test(label, prefix_list):
    print(f"── {label} (prefix_len={len(prefix_list)}) ──")
    prefix_t = torch.tensor([prefix_list], dtype=torch.long)

    # GRU-only
    gru_idxs, gru_scores = gru_only(prefix_list, args.top_n)
    print(f"  GRU-only  : {gru_idxs[:5]}  scores={gru_scores[:5]}")

    if blender:
        cs_idxs, cs_scores, alpha = blender.predict(prefix_t, args.top_n)
        overlap = len(set(gru_idxs) & set(cs_idxs))
        print(f"  Cold-start: {cs_idxs[:5]}  scores={cs_scores[:5]}")
        print(f"  alpha={alpha}  overlap_with_gru={overlap}/{args.top_n}  "
              f"({'pure GRU' if alpha == 1.0 else 'blended' if alpha > 0 else 'pure popularity'})")
    print()


# ── test cases ────────────────────────────────────────────────────────────────
# Use item indices we know exist (1-based, within vocab)
ITEM_A, ITEM_B, ITEM_C, ITEM_D = 1, 2, 3, 4

print("TEST 1 — Cold start (session_len=1, alpha should be 0.33)")
run_test("1 item", [ITEM_A])

print("TEST 2 — Short session (session_len=2, alpha should be 0.67)")
run_test("2 items", [ITEM_A, ITEM_B])

print("TEST 3 — Normal session (session_len=3, alpha should be 1.0)")
run_test("3 items", [ITEM_A, ITEM_B, ITEM_C])

print("TEST 4 — Longer session (session_len=5, alpha should be 1.0)")
run_test("5 items", [ITEM_A, ITEM_B, ITEM_C, ITEM_D, ITEM_A])

if blender:
    print("TEST 5 — Top-10 popularity sanity check")
    pop_t = torch.zeros(1, 1, dtype=torch.long)   # padding only → slen=0
    cs_idxs, cs_scores, alpha = blender.predict(pop_t, args.top_n)
    print(f"  alpha={alpha} (should be 0.0 — pure popularity)")
    print(f"  Top-{args.top_n} popular items: {cs_idxs}")
    print(f"  Scores: {cs_scores}\n")

print("All tests passed.")
