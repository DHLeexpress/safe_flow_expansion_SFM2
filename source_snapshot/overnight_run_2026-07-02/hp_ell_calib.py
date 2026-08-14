"""ell CALIBRATION for the GP-RBF σ (user 2026-07-05: 'uncertainty is almost uniform in the tree viz').
At the 3 probe states: 256 temp-1.3 candidates from res2w256_ft; buffer = 512 dataset windows (the tree-viz
stand-in); sweep ell and report σ mean/std/range → pick ell* = argmax within-node σ-std (discrimination).
Also prints the raw feature distances (cand↔cand, cand↔buffer) and the median-heuristic ell."""
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import grid_scene as GS
import hp_mm_check as MM
import hp_arch_sweep as ARCH
import hp_tree_viz as TV
from uncertainty import GPUncertainty

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--temp", type=float, default=2.0)
_A = _ap.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "hp_test")

pol, _ = ARCH.load_arch("results/hp_arch/res2w256_ft.pt", device=DEV)
env = GS.make_grid()
states, probes = MM.probe_states(env)
buf = TV.dataset_buffer(pol)
Bn = torch.nn.functional.normalize(buf, dim=1)
ELLS = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
meds = []
feats = []
torch.manual_seed(0)
for (t, g, l, h) in probes:
    gt, lt, ht = (torch.tensor(np.asarray(x), device=DEV) for x in (g, l, h))
    with torch.no_grad():
        Uc = pol.sample_window(gt, lt, ht, n=256, temp=_A.temp, nfe=8)
        F = pol.phi_s_at(Uc, gt, lt, ht, s=0.9)
    feats.append((t, F))
    Fn = torch.nn.functional.normalize(F, dim=1)
    iu = torch.triu_indices(256, 256, 1)
    dcc = float(torch.cdist(Fn, Fn)[iu[0], iu[1]].median())
    dcb = float(torch.cdist(Fn, Bn).min(1).values.median())
    meds.append((t, dcc, dcb))
    print(f"probe t={t}: cand↔cand median {dcc:.3f} · cand↔buffer nearest median {dcb:.3f}", flush=True)
ell_med = float(np.median([m[2] for m in meds]))
ELLS_ALL = sorted(set(ELLS + [round(ell_med, 3), round(ell_med / np.sqrt(2), 3)]))
tab = {}
for e in ELLS_ALL:
    unc = GPUncertainty(kernel="rbf", lengthscale=e, lam=1e-2, normalize=True)
    unc.set_buffer(buf)
    stds = []
    for (t, F) in feats:
        s = unc.sigma(F)
        s = (s.detach().cpu().numpy() if torch.is_tensor(s) else np.asarray(s)).reshape(-1)
        stds.append((float(s.mean()), float(s.std()), float(s.max() - s.min())))
    tab[e] = stds
    print(f"ell {e:6.3f}: σ-std/probe {[f'{x[1]:.4f}' for x in stds]} mean-std "
          f"{np.mean([x[1] for x in stds]):.4f} (σ-mean {[f'{x[0]:.3f}' for x in stds]})", flush=True)
best = max(tab, key=lambda e: np.mean([x[1] for x in tab[e]]))
print(f"ELL* = {best}  (median-heuristic dcb = {ell_med:.3f})", flush=True)
fig, ax = plt.subplots(figsize=(8, 5))
for pi in range(len(feats)):
    ax.plot(ELLS_ALL, [tab[e][pi][1] for e in ELLS_ALL], "-o", label=f"probe t={meds[pi][0]}")
ax.axvline(best, color="#d62728", ls="--", label=f"ell*={best}")
ax.set_xscale("log"); ax.set_xlabel("ell (RBF lengthscale)"); ax.set_ylabel("within-node σ std")
ax.set_title(f"ell calibration — σ discrimination over 256 candidates/probe (res2w256_ft, temp {_A.temp})")
ax.legend(); ax.grid(alpha=.3)
fig.savefig(os.path.join(FIG, "ell_calibration.png"), dpi=125, bbox_inches="tight")
print("saved figures/hp_test/ell_calibration.png", flush=True)
