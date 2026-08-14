"""HP-chessboard saturation plot from the run log — same 4-panel layout as figures/reduced_henc_trend.png
(the ctx=48 collapse exemplar) so the contrast reads at a glance. HP-only per user (no overlaid comparison)."""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse as _arg
_p = _arg.ArgumentParser(); _p.add_argument("--log", default="/home/dohyun/.claude/jobs/cab73065/tmp/hp_full.log")
_p.add_argument("--out", default="hp_full20k_trend.png"); _p.add_argument("--title", default="")
_A = _p.parse_args()
LOG = _A.log
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

pat = re.compile(r"it(\d+): val2 (\d+)% \(γ:(\d+)/(\d+)/(\d+)\) cov_cum ([\d.]+)% cov_fin ([\d.]+)% varσ ([\d.]+) "
                 r"viol\[task (\d+) appr (\d+) socp (\d+)\] drift ([\d.]+) demoCFM ([\d.]+)")
rows = [pat.match(l).groups() for l in open(LOG) if pat.match(l)]
it = [int(r[0]) for r in rows]
v = [int(r[1]) for r in rows]
g5, g10, g1 = ([int(r[i]) for r in rows] for i in (2, 3, 4))   # cfg.gammas ORDER = (0.5, 1.0, 0.1)!
cc = [float(r[5]) for r in rows]
socp = [int(r[10]) for r in rows]
dr = [float(r[11]) for r in rows]
dc = [float(r[12]) for r in rows]

fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
ax[0].plot(it, v, "-o", color="#2ca02c", lw=2, label="γ-mean")
for ys, c, lb in ((g1, "#1f77b4", "γ0.1"), (g5, "#ff7f0e", "γ0.5"), (g10, "#d62728", "γ1.0")):
    ax[0].plot(it, ys, "--", color=c, alpha=.6, lw=1.2, label=lb)
ax[0].set_title("A) validity2 % — 20k iters NO COLLAPSE; γ1.0 consolidates ~64-84, γ0.1 starves"); ax[0].legend(fontsize=8)
ax[1].plot(it, cc, "-o", color="#1f77b4", lw=2)
ax[1].set_title("B) coverage_cumulative % — 100/252 staircases discovered (39.8%)")
ax[2].plot(it, socp, "-o", color="#d62728", lw=2)
ax[2].set_title("C) SOCP violation %")
ax[3].plot(it, dr, "-o", color="#9467bd", lw=2, label="E_hp ctx drift")
ax3b = ax[3].twinx(); ax3b.plot(it, dc, "-s", color="#8c564b", lw=1.5, label="demo-CFM")
ax[3].set_title("D) drift plateaus at 0.757 from ~it8000 (stable attractor)")
ax[3].legend(loc="center right", fontsize=8); ax3b.legend(loc="lower right", fontsize=8)
for a in ax:
    a.grid(alpha=.25); a.set_xlabel("iteration")
fig.suptitle("H_P inductive-bias reduced model (ctx = raw5 ⊕ E_hp(H_P)→32, 101k params) — positive-only expansion "
             "2000 iters, 0702 defaults: validity2 SATURATES instead of collapsing", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIG, _A.out), dpi=125, bbox_inches="tight")
print("saved", _A.out, flush=True)
