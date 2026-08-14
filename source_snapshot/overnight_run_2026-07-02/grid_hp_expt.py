"""H_P inductive-bias experiment on the 0702 CHESSBOARD (user 2026-07-04 spec, verbatim):
  model ctx = raw low5(5) ⊕ E_g([1,16,12] H_P channel → shallow CNN → AdaptiveAvgPool → 32);
  trunk input = [U(20) + ctx(37) + fourier-t(32)]. NO E_l mixing the raw conditions, NO hist, NO GRU.
Hypothesis: the reduced models' validity JIGGLED because ctx was OOD each iteration and the optimizer remaps the
encoder instead of refining p(U|ctx); the H_P grid channel is the inductive bias that should let coverage(252)
and validity2 SATURATE. Protocol: pretrain on the 0702 demo windows, then `grid_expand2.run_expand2` UNTOUCHED
(validity2 gate + 252 coverage + varσ + probes, SFG2Config defaults, iters=2000 as run_reduced_0703). No control
arm (user: don't compare — watch reliability of coverage/validity2/varσ intermittently)."""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

try:
    import _paths  # noqa: F401  (0704 bootstrap; not needed when run from 0702)
except ImportError:
    pass
import grid_policy2 as GP2
import grid_expand2 as GX2
import grid_scene as GS
import wandb_utils as W

HERE = os.path.dirname(os.path.abspath(__file__))
R2 = os.path.join(os.path.dirname(HERE), "overnight_run_2026-07-02")
OUT = os.path.join(HERE, "results", "hp_chessboard")
os.makedirs(OUT, exist_ok=True)


class GridHPFlowPolicy(GP2.GridGRUFlowPolicy2):
    """ctx = raw low5(5) ⊕ E_g(H_P[1,16,12]→32). Slices channel 2 (clipped H_P) from the standard 3-ch grid."""
    def __init__(self, width=256, depth=2, u_max=1.0, use_gru=False, **kw):
        super().__init__(grid_shape=(1, 16, 12), width=width, depth=depth, u_max=u_max,
                         use_gru=use_gru, encode_low=False, use_grid=True, **kw)
        # shallow 1-ch CNN + AdaptiveAvgPool → 32 (parent pattern, half the params, H_P only)
        self.enc_grid = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.SiLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 3)), nn.Flatten(),
            nn.Linear(16 * 4 * 3, 32), nn.SiLU())
        gd = self.gru_dim if use_gru else 0                      # GRU(16) over past controls → curvature (2026-07-06)
        self.ctx_dim = 5 + gd + 32                               # raw low5 (+ GRU token if on) + H_P token
        in_dim = self.d + self.ctx_dim + self.t_dim              # 20 + 37 + 32 = 89 (105 with GRU)
        layers = [nn.Linear(in_dim, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        self.trunk = nn.Sequential(*layers)                       # head (width→20) unchanged

    def ctx_from(self, grid, low5, hist):
        hp = grid[..., 2:3, :, :]                                 # H_P channel from the standard [.,3,16,12]
        return super().ctx_from(hp, low5, hist)

    def config(self):
        return dict(arch="hp-reduced-32", H_pred=self.H_pred, grid_shape=(1, 16, 12), K_hist=self.K_hist,
                    width=self.width, depth=2, u_max=self.u_max, ctx_dim=self.ctx_dim)


def save_hp(policy, path, extra=None):
    d = {"state_dict": policy.state_dict(), "config": policy.config()}
    if extra:
        d.update(extra)
    torch.save(d, path)


def load_hp(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    pol = GridHPFlowPolicy(width=ck["config"]["width"], u_max=ck["config"]["u_max"])
    pol.load_state_dict(ck["state_dict"])
    return pol.to(device).eval(), ck


def pretrain(dev, epochs=120, batch=256, lr=3e-4, warmup=5):
    G, L, Hh, U = [], [], [], []
    for g in ("0.1", "0.5", "1.0"):
        d = torch.load(os.path.join(R2, "dataset", f"windows_g{g}.pt"))
        G.append(d["grid"]); L.append(d["low5"]); Hh.append(d["hist"]); U.append(d["U"])
    G, L, Hh, U = (torch.cat(x) for x in (G, L, Hh, U))
    n = G.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    G, L, Hh, U = G[perm], L[perm], Hh[perm], U[perm]
    nval = max(2048, n // 10)
    tr = TensorDataset(G[nval:], L[nval:], Hh[nval:], U[nval:])
    va = (G[:nval].to(dev), L[:nval].to(dev), Hh[:nval].to(dev), U[:nval].to(dev))
    dl = DataLoader(tr, batch_size=batch, shuffle=True, drop_last=True)
    pol = GridHPFlowPolicy().to(dev)
    npar = sum(p.numel() for p in pol.parameters())
    print(f"[pretrain] {n} windows ({n-nval}/{nval}) · HP model {npar/1e3:.1f}k params "
          f"(E_hp {sum(p.numel() for p in pol.enc_grid.parameters())/1e3:.1f}k)", flush=True)
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: (ep + 1) / warmup if ep < warmup else
        0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(1, epochs - warmup))))
    best = (float("inf"), None)
    for ep in range(epochs):
        pol.train()
        tot = nb = 0
        for gb, lb, hb, ub in dl:
            loss = pol.cfm_loss(ub.to(dev), pol.ctx_from(gb.to(dev), lb.to(dev), hb.to(dev)))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        sched.step()
        pol.eval()
        with torch.no_grad():
            torch.manual_seed(0)
            v = float(pol.cfm_loss(va[3], pol.ctx_from(va[0], va[1], va[2])))
        if v < best[0]:
            best = (v, {k: x.detach().cpu().clone() for k, x in pol.state_dict().items()})
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"[pretrain] ep {ep:03d} train {tot/nb:.4f} val {v:.4f}", flush=True)
    pol.load_state_dict(best[1])
    save_hp(pol, os.path.join(OUT, "pretrained_hp.pt"), extra={"best_val": best[0]})
    print(f"[pretrain] saved pretrained_hp.pt (val {best[0]:.4f})", flush=True)
    return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)            # run_reduced_0703 protocol; FULL run = 20000
    ap.add_argument("--skip-pretrain", action="store_true")
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--name", default="hp-chessboard")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--enc-lr-mult", type=float, default=None)
    ap.add_argument("--inner-steps", type=int, default=None)
    ap.add_argument("--ell", type=float, default=None)
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--measure-every", type=int, default=None)
    ap.add_argument("--n-measure", type=int, default=None)
    ap.add_argument("--s", type=float, default=None)
    ap.add_argument("--grad-clip", type=float, default=None)
    ap.add_argument("--ckpt-every", type=int, default=None)
    ap.add_argument("--demo-frac", type=float, default=None)
    ap.add_argument("--lwf-eta", type=float, default=None)
    ap.add_argument("--arch-ckpt", default=None, help="start from an hp_arch checkpoint (ResTrunk-aware)")
    W.add_wandb_args(ap)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)
    if args.arch_ckpt:
        import hp_arch_sweep as ARCH
        pol, _ = ARCH.load_arch(args.arch_ckpt, device=dev)
        print(f"[main] loaded arch ckpt {args.arch_ckpt}", flush=True)
    elif args.skip_pretrain and os.path.exists(os.path.join(OUT, "pretrained_hp.pt")):
        pol, _ = load_hp(os.path.join(OUT, "pretrained_hp.pt"), device=dev)
        print("[main] loaded existing pretrained_hp.pt", flush=True)
    else:
        pol = pretrain(dev)
    env = GS.make_grid()
    cfg = GX2.SFG2Config(iters=args.iters)                        # 0702 defaults; sweep overrides only if given
    for k in ("lr", "alpha", "beta", "enc_lr_mult", "inner_steps", "ell", "temp", "measure_every", "n_measure", "demo_frac", "lwf_eta", "s", "grad_clip", "ckpt_every"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
            print(f"[main] override {k}={v}", flush=True)
    run = W.init_run(args, name=args.name, config={**vars(args), **pol.config()}, group="sfm-0704")
    print(f"[main] EXPANSION: iters={cfg.iters} temp={cfg.temp} ell={cfg.ell} s={cfg.s} beta={cfg.beta} "
          f"N={cfg.N} gp_buf={cfg.gp_buf} (positive-only, validity2 gate, 252 coverage)", flush=True)
    GX2.run_expand2(pol, env, cfg, device=dev, run=run, outdir=args.outdir, log=print)
    W.finish(run)


if __name__ == "__main__":
    main()
