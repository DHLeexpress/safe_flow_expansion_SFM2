"""HP step 0.2/0.3 — arch sweep to find the best 89-in→20-out model (user 2026-07-05): trunk variants
{d2w256 (baseline), d3w256, res2w256, res3w256 (pre-LN ResNet-MLP, the proven 0704 recipe), d2w384} trained on
the enlarged dataset, then AUTO-EVAL: val-cfm + validity2@iter0 (n=25/γ, temp 1.0, the standard) + multimodality
splits at the 3 probe states (hp_mm_check). Usage: python hp_arch_sweep.py --variant res2w256"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import grid_scene as GS
import grid_rollout as GR
import grid_metrics2 as GM2
import grid_hp_expt as HP
import hp_mm_check as MM

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "hp_arch")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

VARIANTS = {
    "d2w256": dict(depth=2, width=256, res=0),
    "d3w256": dict(depth=3, width=256, res=0),
    "res2w256": dict(depth=2, width=256, res=2),
    "res3w256": dict(depth=2, width=256, res=3),
    "d2w384": dict(depth=2, width=384, res=0),
    "res2w256_gru": dict(depth=2, width=256, res=2, use_gru=True),   # GRU over past controls (2026-07-06)
}


class ResTrunk(nn.Module):
    """Input proj + n pre-LN residual MLP blocks (0704 cooked-trunk recipe). Output dim = width (= φ_s)."""
    def __init__(self, in_dim, width, n_blocks, dropout=0.05):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, width), nn.SiLU())
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(),
                                                   nn.Linear(width, width), nn.Dropout(dropout))
                                     for _ in range(n_blocks)])

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return h


def build(name):
    v = VARIANTS[name]
    pol = HP.GridHPFlowPolicy(width=v["width"], depth=v["depth"], use_gru=v.get("use_gru", False))
    if v["res"] > 0:
        pol.trunk = ResTrunk(pol.d + pol.ctx_dim + pol.t_dim, v["width"], v["res"])
    return pol


def train(pol, name, epochs=120, batch=256, lr=3e-4, warmup=5):
    G, L, Hh, U = [], [], [], []
    for g in ("0.1", "0.5", "1.0"):
        d = torch.load(os.path.join(HERE, "dataset", f"windows_g{g}.pt"))
        G.append(d["grid"]); L.append(d["low5"]); Hh.append(d["hist"]); U.append(d["U"])
    G, L, Hh, U = (torch.cat(x) for x in (G, L, Hh, U))
    n = G.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    G, L, Hh, U = G[perm], L[perm], Hh[perm], U[perm]
    nval = max(4096, n // 10)
    tr = TensorDataset(G[nval:], L[nval:], Hh[nval:], U[nval:])
    va = (G[:nval].to(DEV), L[:nval].to(DEV), Hh[:nval].to(DEV), U[:nval].to(DEV))
    dl = DataLoader(tr, batch_size=batch, shuffle=True, drop_last=True)
    pol = pol.to(DEV)
    npar = sum(p.numel() for p in pol.parameters())
    print(f"[{name}] {n} windows ({n-nval} train) · {npar/1e3:.1f}k params", flush=True)
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: (ep + 1) / warmup if ep < warmup else
                                              0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(1, epochs - warmup))))
    hist = []
    best = (float("inf"), None, -1)
    for ep in range(epochs):
        pol.train()
        tot = nb = 0
        for gb, lb, hb, ub in dl:
            loss = pol.cfm_loss(ub.to(DEV), pol.ctx_from(gb.to(DEV), lb.to(DEV), hb.to(DEV)))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        sched.step()
        pol.eval()
        with torch.no_grad():
            torch.manual_seed(0)
            v = float(pol.cfm_loss(va[3], pol.ctx_from(va[0], va[1], va[2])))
        hist.append(dict(ep=ep, train=tot / nb, val=v))
        if v < best[0]:
            best = (v, {k: x.detach().cpu().clone() for k, x in pol.state_dict().items()}, ep)
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"[{name}] ep {ep:03d} train {tot/nb:.4f} val {v:.4f}", flush=True)
    pol.load_state_dict(best[1])
    torch.save({"state_dict": pol.state_dict(), "variant": name, "best_val": best[0], "best_ep": best[2]},
               os.path.join(OUT, f"{name}.pt"))
    json.dump(hist, open(os.path.join(OUT, f"{name}_curve.json"), "w"))
    return pol, best[0]


def evaluate(pol, name):
    env = GS.make_grid()
    gl = env.goal.detach().cpu().numpy()
    val = {}
    for g in (0.1, 0.5, 1.0):
        torch.manual_seed(1)
        paths = GR.deploy_many(pol, env, g, 25, T=250, temp=1.0, nfe=8, device=DEV)
        ok = reach = 0
        for p in paths:
            P = np.asarray(p, np.float32)
            reach += int(np.linalg.norm(P[-1, :2] - gl) < 0.5)
            v = GM2.traj_valid2(P, env, g)
            ok += int(v[0] if isinstance(v, tuple) else bool(v))
        val[g] = (ok * 4, reach * 4)
    states, probes = MM.probe_states(env)
    obs = env.obstacles.detach().cpu().numpy()
    mm = []
    for (t, g, l, h) in probes:
        gt, lt, ht = (torch.tensor(np.asarray(x), device=DEV) for x in (g, l, h))
        with torch.no_grad():
            U = pol.sample_window(gt, lt, ht, n=256, temp=1.0, nfe=8).detach().cpu().numpy()
        rolls = GR.di_rollout_batch(states[t], U, 0.1)
        sc, _, _ = MM.lateral_split(states[t], rolls[:, -1], obs)
        mm.append(round(sc * 100))
    print(f"RESULT {name}: validity2@it0 " +
          " ".join(f"γ{g}:{val[g][0]}%/{val[g][1]}%r" for g in (0.1, 0.5, 1.0)) +
          f" | mm-splits {mm} (start/enc/mid)", flush=True)
    return val, mm


def load_arch(path, device="cpu"):
    """Load an arch-sweep checkpoint (rebuilds ResTrunk variants that HP.load_hp cannot)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    pol = build(ck["variant"])
    pol.load_state_dict(ck["state_dict"])
    return pol.to(device).eval(), ck


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    a = ap.parse_args()
    pol = build(a.variant)
    pol, bv = train(pol, a.variant)
    val, mm = evaluate(pol, a.variant)
    json.dump(dict(variant=a.variant, best_val=bv, validity=str(val), mm=mm),
              open(os.path.join(OUT, f"{a.variant}_result.json"), "w"))
    print(f"[{a.variant}] DONE best_val {bv:.4f}", flush=True)
