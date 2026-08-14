"""Windowed MPPI-expert dataset (SOTA diffusion/flow-matching policy style).

One training sample = one executed step of a SafeMPPI rollout:
    input  = (polar polytope-occupancy grid [3,16,12], low-dim goal-aligned state [7], gamma)
    target = the MPPI reward-weighted planned control window U_{t:t+H_pred-1}, in the goal-aligned
             LOCAL frame  U_local [H_pred, 2].
So N episodes × T steps become N·T windows — small per-sample dimension (raises verifier validity vs a
single 80×2 one-shot). Reusable Dataset/collate + save/load; the builder lives in the run folder.
"""
from __future__ import annotations

import os

import torch
from torch.utils.data import Dataset

KEYS = ("grid", "low_dim", "gamma", "U_local")


class WindowedDataset(Dataset):
    def __init__(self, path):
        d = torch.load(path, weights_only=False)
        self.grid = d["grid"].float()             # [N,3,Nθ,Nr]
        self.low_dim = d["low_dim"].float()       # [N,7]
        self.gamma = d["gamma"].float()           # [N]
        self.U_local = d["U_local"].float()       # [N,H_pred,2]
        self.meta = d.get("meta", {})

    def __len__(self):
        return self.grid.shape[0]

    def __getitem__(self, i):
        return {"grid": self.grid[i], "low_dim": self.low_dim[i],
                "gamma": self.gamma[i], "U_local": self.U_local[i]}


def windowed_collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def _split_indices(n, splits, seed=0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_tr = int(splits[0] * n)
    n_va = int(splits[1] * n)
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]


def save_windowed_splits(out_dir, grid, low_dim, gamma, U_local, meta=None,
                         splits=(0.8, 0.1, 0.1), seed=0):
    os.makedirs(out_dir, exist_ok=True)
    n = grid.shape[0]
    tr, va, te = _split_indices(n, splits, seed)
    for name, idx in [("train", tr), ("val", va), ("test", te)]:
        torch.save({"grid": grid[idx], "low_dim": low_dim[idx], "gamma": gamma[idx],
                    "U_local": U_local[idx], "meta": meta or {}},
                   os.path.join(out_dir, f"{name}.pt"))
    return {"n_total": int(n), "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te))}
