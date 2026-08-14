"""QUOTA-experiment report figure (user 2026-07-05): coverage-validity trend every 100 iters (initial point =
pretrained) + new-id registration timeline + buffer/ban statistics — the companion to the N-row tree.
Usage: python hp_quota_viz.py --outdir results/hp_quota --tag quota15"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "hp_test")
GAMMAS = ("0.5", "1.0", "0.1")
COLS = {"0.5": "#ff7f0e", "1.0": "#d62728", "0.1": "#1f77b4"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/hp_quota")
    ap.add_argument("--tag", default="quota15")
    a = ap.parse_args()
    d = json.load(open(os.path.join(a.outdir, "history.json")))
    hist = d["hist"]
    it = [0] + [r["it"] for r in hist]
    pre = d.get("pre_val2", {})
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))

    for g in GAMMAS:
        v0 = pre.get(g, hist[0]["val2"][g]) * 100
        ax[0].plot(it, [v0] + [r["val2"][g] * 100 for r in hist], "-o", ms=4, color=COLS[g], label=f"γ{g}")
    vm = [np.mean([pre.get(g, 0) for g in GAMMAS]) * 100] + \
         [np.mean([r["val2"][g] for g in GAMMAS]) * 100 for r in hist]
    ax[0].plot(it, vm, "-", color="k", lw=2.4, alpha=.6, label="γ-mean")
    ax[0].set_title("validity2 % every 100 it (it0 = pretrained)"); ax[0].legend(fontsize=8)

    nr = [0] + [r.get("new_registered", r.get("new_ids", 0)) for r in hist]
    ax[1].plot(it, nr, "-o", ms=4, color="#2ca02c", lw=2)
    ax[1].set_title("COVERAGE: cumulative NEW staircase-ids registered")

    ax[2].plot(it, [0] + [r["n"] for r in hist], "-o", ms=4, color="#4c72b0", label="buffer windows")
    ax2b = ax[2].twinx()
    ax2b.plot(it, [0] + [r["banned"] for r in hist], "-s", ms=4, color="#d62728", label="banned ids")
    ax2b.plot(it, [0] + [r["new_ids"] for r in hist], "-^", ms=4, color="#2ca02c", label="new ids in buffer")
    ax[2].set_title("buffer size · bans · new-id strata")
    ax[2].legend(loc="upper left", fontsize=8); ax2b.legend(loc="upper right", fontsize=8)

    dr = [0] + [r["droughts"] for r in hist]
    up = [0] + [r["updates"] for r in hist]
    ax[3].plot(it, dr, "-o", ms=4, color="#8c564b", label="droughts (wait)")
    ax[3].plot(it, up, "-o", ms=4, color="#1f77b4", label="updates")
    ax[3].set_title("WAIT-mode: droughts vs updates (cumulative)"); ax[3].legend(fontsize=8)
    for A in ax:
        A.grid(alpha=.25); A.set_xlabel("iteration")
    fig.suptitle(f"QUOTA EXPANSION [{a.tag}] — staircase-id quotas (≤5% existing · new immune 100 it · "
                 "10-streak ban · Hamming-1 inherits · per-γ books)", fontsize=12.5)
    fig.tight_layout()
    out = os.path.join(FIG, f"quota_trend_{a.tag}.png")
    fig.savefig(out, dpi=125, bbox_inches="tight")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
