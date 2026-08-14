"""HP step 0.1 — MORE demo data (user 2026-07-05): seeds 150..599 (the stock script's --append would DUPLICATE
seeds 0-149; this wrapper offsets correctly), merged into dataset/windows_g{γ}.pt. Backup kept in dataset/backup_450traj/."""
import os
import shutil
import time

import torch

import stage2_grid_data as SD
import grid_scene as GS

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
BAK = os.path.join(DATA, "backup_450traj")
os.makedirs(BAK, exist_ok=True)

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--s0", type=int, default=150); _ap.add_argument("--s1", type=int, default=600)
_a = _ap.parse_args()
S0, S1 = _a.s0, _a.s1
env = GS.make_grid()
cfg = GS.mode1_config()
for g in (0.1, 0.5, 1.0):
    out = os.path.join(DATA, f"windows_g{g}.pt")
    if not os.path.exists(os.path.join(BAK, f"windows_g{g}.pt")):
        shutil.copy(out, BAK)
    G, L, Hh, U = [], [], [], []
    n_ok = 0
    t0 = time.time()
    for s in range(S0, S1):
        states, controls = SD.rollout_full(env, g, cfg, s)
        ok, _ = GS.is_success(states[:, :2], env)
        if not ok or len(controls) < 2:
            continue
        n_ok += 1
        gg, ll, hh, uu = SD.windows_from(states, controls, env, g)
        G += gg; L += ll; Hh += hh; U += uu
        if (s - S0 + 1) % 50 == 0:
            print(f"  γ{g}: {s-S0+1}/{S1-S0} seeds, {n_ok} success, {len(G)} windows, "
                  f"{(time.time()-t0)/(s-S0+1):.2f}s/seed", flush=True)
    d = torch.load(out)
    add = dict(grid=torch.tensor(G), low5=torch.tensor(L), hist=torch.tensor(Hh), U=torch.tensor(U))
    for k in ("grid", "low5", "hist", "U"):
        d[k] = torch.cat([d[k], torch.as_tensor(add[k], dtype=d[k].dtype)], 0)
    d["n_traj"] = int(d.get("n_traj", 0)) + n_ok
    d["n_seeds"] = int(d.get("n_seeds", 0)) + (S1 - S0)
    torch.save(d, out)
    print(f"γ{g}: +{n_ok} trajs / +{len(G)} windows → total {d['grid'].shape[0]} windows, {d['n_traj']} trajs", flush=True)
print("DONE", flush=True)
