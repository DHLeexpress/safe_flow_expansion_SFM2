"""s CALIBRATION (user 2026-07-06: "set s to increase the uncertainty distribution spread, like we did for ell").
s is the flow-interpolation level at which phi_s probes the field: x_s=(1-s)*x0 + s*x1, tau=s (flow_policy.phi_s).
s->1 => x_s == the actual (diverse) candidate controls => features spread => GP-sigma discriminates.
s->0 => x_s == shared noise templates => features collapse => sigma ~uniform.
At the 3 probe states: 256 candidates from res2w256_ft_v2, buffer=512 dataset windows, ell fixed at 0.5;
sweep s and report sigma mean/std/range -> pick s* = argmax within-node sigma-std."""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths  # noqa: F401
import grid_scene as GS
import hp_mm_check as MM
import hp_arch_sweep as ARCH
import hp_tree_viz as TV
from uncertainty import GPUncertainty

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="results/hp_arch/res2w256_ft_v2.pt")
ap.add_argument("--ell", type=float, default=0.5)
ap.add_argument("--temp", type=float, default=1.5)
A = ap.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "dr_test_overnight")
os.makedirs(FIG, exist_ok=True)

pol, _ = ARCH.load_arch(A.ckpt, device=DEV)
env = GS.make_grid()
states, probes = MM.probe_states(env)
SS = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]
torch.manual_seed(0)

# cache candidate controls per probe (sampling is s-independent; only phi_s depends on s)
cand = []
for (t, g, l, h) in probes:
    gt, lt, ht = (torch.tensor(np.asarray(x), device=DEV) for x in (g, l, h))
    with torch.no_grad():
        Uc = pol.sample_window(gt, lt, ht, n=256, temp=A.temp, nfe=8)
    cand.append((t, gt, lt, ht, Uc))

tab = {}
for s in SS:
    stds, means = [], []
    unc = GPUncertainty(kernel="rbf", lengthscale=A.ell, lam=1e-2, normalize=True)
    unc.set_buffer(TV.dataset_buffer(pol, s=s))            # buffer featurized at the SAME s (fair kernel)
    for (t, gt, lt, ht, Uc) in cand:
        with torch.no_grad():
            F = pol.phi_s_at(Uc, gt, lt, ht, s=s)
        sig = unc.sigma(F)
        sig = (sig.detach().cpu().numpy() if torch.is_tensor(sig) else np.asarray(sig)).reshape(-1)
        stds.append(float(sig.std())); means.append(float(sig.mean()))
    tab[s] = (means, stds)
    print(f"s {s:.2f}: sigma-std/probe {[f'{x:.4f}' for x in stds]} mean-std {np.mean(stds):.4f} "
          f"(sigma-mean {[f'{x:.3f}' for x in means]})", flush=True)

best = max(tab, key=lambda s: np.mean(tab[s][1]))
print(f"S* = {best}  (max within-node sigma-std = {np.mean(tab[best][1]):.4f})", flush=True)
fig, ax = plt.subplots(figsize=(8, 5))
for pi in range(len(cand)):
    ax.plot(SS, [tab[s][1][pi] for s in SS], "-o", label=f"probe t={cand[pi][0]}")
ax.axvline(best, color="#d62728", ls="--", label=f"s*={best}")
ax.set_xlabel("s (flow-interpolation level for phi_s)"); ax.set_ylabel("within-node sigma std")
ax.set_title(f"s calibration - sigma spread over 256 candidates/probe (res2w256_ft_v2, ell {A.ell})")
ax.legend(); ax.grid(alpha=.3)
fig.savefig(os.path.join(FIG, "s_calibration.png"), dpi=125, bbox_inches="tight")
print("saved figures/dr_test_overnight/s_calibration.png", flush=True)
