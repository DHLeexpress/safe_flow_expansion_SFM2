"""Constructive sweep orchestrator (2026-07-03, user's plan §4 + entanglement/validity arms).

P0  pretrain2 widths {256,192,128} (concurrent) -> pick lightest within tolerance     -> model_choice.json
S1  lr    {2e-4°, 1e-4, 1e-5}
S2  enc   {update-all°, freeze_enc}          (causal entanglement arm)
S3  alpha {0°, 0.005, 0.01}
S4  beta  {1/10°, 1/5, 1/20}
S5  s     {0.9°, 0.8, 0.3}
S6  pos_margin {0°, 0.03}                    (validity data-hygiene arm)
Each stage fixes previous winners (° = incumbent, reused not re-run). 2k iters per config, all on ONE GPU
(CUDA_VISIBLE_DEVICES set by the caller), ≤3 workers concurrent. Scoring per run (γ-averaged, iter-0 excluded):
  0.4·mean validity2 + 0.3·final coverage_cum + 0.2·mean(last-3 coverage_final)/(n_measure/252)
  + 0.1·mean var(σ) (max-normalized within the comparison set).
Everything resumable: cached run dirs / pretrain jsons are skipped. Master appends achievements to
GOAL_07_02.md (append-only) and finishes with overlay + entanglement figures, a 100-deploy/γ final eval of
the winner, and the two videos (grid_expand_viz2.py).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "results", "sweep_0703")
FIG = os.path.join(HERE, "figures")
GOAL_MD = os.path.join(HERE, "GOAL_07_02.md")
PY = sys.executable
WIDTHS = [256, 192, 128]
GAMMAS = [0.5, 1.0, 0.1]
N_MEASURE = 25
STAGE_DEADLINE_S = 12 * 3600
# Positive-only (no U_demo). MAIN path = GP-RBF + online. enc_mult = encoder/field lr ratio (the "nice
# optimizer" lever for the entangled representation; 0.0 == frozen encoder). NN + round are PARALLEL BACKUPS.
BASE = dict(lr=2e-4, enc_mult=1.0, alpha=0.0, beta=0.1, s=0.9, margin=0.0, unc="gp", schedule="online")


def ts():
    return time.strftime("%H:%M:%S")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


def goal_append(text):
    with open(GOAL_MD, "a") as f:
        f.write(text if text.endswith("\n") else text + "\n")


def run_id(c):
    rid = f"lr{c['lr']:g}_a{c['alpha']:g}_b{c['beta']:g}_s{c['s']:g}"
    if c.get("enc_mult", 1.0) != 1.0:
        rid += f"_e{c['enc_mult']:g}"
    if c["margin"] > 0:
        rid += f"_m{c['margin']:g}"
    if c.get("unc", "gp") != "gp":
        rid += f"_{c['unc']}"
    if c.get("schedule", "online") != "online":
        rid += f"_{c['schedule']}"
    return rid


def build_cmd(c, rid, outdir, width):
    cmd = [PY, "grid_expand2.py", "--policy", f"pretrained2_w{width}.pt", "--outdir", outdir,
           "--run-id", rid, "--iters", "2000", "--lr", f"{c['lr']:g}", "--alpha", f"{c['alpha']:g}",
           "--beta", f"{c['beta']:g}", "--s", f"{c['s']:g}", "--enc-lr-mult", f"{c.get('enc_mult', 1.0):g}",
           "--n-measure", str(N_MEASURE), "--unc", c.get("unc", "gp"),
           "--schedule", c.get("schedule", "online"), "--wandb-mode", "online"]
    if c["margin"] > 0:
        cmd += ["--pos-margin", f"{c['margin']:g}"]
    return cmd


def spawn(cmd, logfile):
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "3")
    env["OMP_NUM_THREADS"] = "8"
    env["LD_LIBRARY_PATH"] = "/home/dohyun/miniforge3/lib:" + env.get("LD_LIBRARY_PATH", "")
    f = open(logfile, "a")
    return subprocess.Popen(cmd, cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT), f


def wait_all(procs, deadline_s, what):
    t0 = time.time()
    while True:
        alive = [(n, p) for n, p, _ in procs if p.poll() is None]
        if not alive:
            break
        if time.time() - t0 > deadline_s:
            for n, p in alive:
                log(f"DEADLINE: killing {what} worker {n}")
                p.kill()
            break
        time.sleep(30)
    out = {}
    for n, p, f in procs:
        f.close()
        out[n] = (p.returncode == 0)
        if p.returncode != 0:
            log(f"worker {n} FAILED (rc={p.returncode}) — excluded from scoring")
    return out


# --------------------------------------------------------------------- P0 pretrain
def phase_pretrain():
    os.makedirs(ROOT, exist_ok=True)
    choice_f = os.path.join(ROOT, "model_choice.json")
    if os.path.exists(choice_f):
        c = json.load(open(choice_f))
        log(f"P0 cached: width {c['width']}")
        return c["width"]
    procs = []
    for w in WIDTHS:
        if os.path.exists(os.path.join(HERE, "results", "pretrain2", f"w{w}.json")):
            log(f"P0: w{w} cached")
            continue
        cmd = [PY, "stage3_pretrain2.py", "--width", str(w), "--wandb-mode", "online"]
        procs.append((f"w{w}", *spawn(cmd, os.path.join(ROOT, f"pretrain_w{w}.log"))))
        log(f"P0 launched pretrain w{w}")
    if procs:
        wait_all(procs, 4 * 3600, "pretrain")
    stats = {}
    for w in WIDTHS:
        f = os.path.join(HERE, "results", "pretrain2", f"w{w}.json")
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        mval = float(np.mean([d["baseline"][str(g)]["validity"] for g in GAMMAS]))
        stats[w] = dict(val_cfm=d["best_val_cfm"], base_validity=mval, params=d["params"]["total"])
    if not stats:
        raise RuntimeError("no pretrain results")
    best_cfm = min(s["val_cfm"] for s in stats.values())
    best_val = max(s["base_validity"] for s in stats.values())
    elig = [w for w, s in stats.items()
            if s["val_cfm"] <= 1.05 * best_cfm and s["base_validity"] >= best_val - 0.05]
    width = min(elig) if elig else min(stats, key=lambda w: stats[w]["val_cfm"])
    json.dump(dict(width=width, stats=stats), open(choice_f, "w"), indent=2)
    log(f"P0 CHOICE: W{width}  ({json.dumps(stats)})")
    goal_append(f"- **[P0 {time.strftime('%m-%d %H:%M')}] model choice: W{width}** — " +
                "; ".join(f"W{w}: val_cfm {s['val_cfm']:.4f}, baseline-validity2 {s['base_validity']*100:.0f}%, "
                          f"{s['params']:,} params" for w, s in sorted(stats.items())) +
                f". Rule: lightest with val_cfm ≤1.05×best & validity within 5 pts.")
    return width


# --------------------------------------------------------------------- scoring
def load_hist(rid):
    f = os.path.join(ROOT, rid, "history.json")
    if not os.path.exists(f):
        return None
    return json.load(open(f))


def series(h, key):
    return [float(np.mean([r[f"g{g}"][key] for g in GAMMAS])) for r in h]


def score_parts(h):
    recs = [r for r in h if r["iter"] > 0]
    if not recs:
        return None
    val = series(recs, "validity")
    covc = series(recs, "coverage_cum")
    covf = series(recs, "coverage_final")
    vs = [r.get("var_sigma", 0.0) for r in recs]
    ceil = N_MEASURE / 252.0
    return dict(val_mean=float(np.mean(val)),
                val_trend=float(np.mean(val[-2:]) - np.mean(val[:2])),
                cov_cum=covc[-1],
                cov_fin=float(np.mean(covf[-3:]) / ceil),
                var_sig=float(np.mean(vs)),
                val_last=val[-1])


def stage_pick(cands, incumbent_rid):
    parts = {}
    for rid in cands:
        h = load_hist(rid)
        if h:
            p = score_parts(h)
            if p:
                parts[rid] = p
    if not parts:
        return incumbent_rid, {}
    vmax = max(p["var_sig"] for p in parts.values()) or 1.0
    scores = {rid: 0.4 * p["val_mean"] + 0.3 * p["cov_cum"] + 0.2 * p["cov_fin"] + 0.1 * (p["var_sig"] / vmax)
              for rid, p in parts.items()}
    win = max(scores, key=scores.get)
    for rid in sorted(scores, key=scores.get, reverse=True):
        p = parts[rid]
        log(f"   {rid}: score {scores[rid]:.3f} (val {p['val_mean']*100:.0f}%/trend {p['val_trend']*100:+.0f} "
            f"cov {p['cov_cum']*100:.1f}% fin {p['cov_fin']*100:.0f}%norm varσ {p['var_sig']:.4f})")
    return win, {rid: dict(score=scores[rid], **parts[rid]) for rid in scores}


# --------------------------------------------------------------------- sweep
def launch_stage(name, cfgs, width):
    procs = []
    for c in cfgs:
        rid = run_id(c)
        outdir = os.path.join(ROOT, rid)
        if os.path.exists(os.path.join(outdir, "history.json")):
            log(f"{name}: {rid} cached")
            continue
        os.makedirs(outdir, exist_ok=True)
        procs.append((rid, *spawn(build_cmd(c, rid, outdir, width), os.path.join(outdir, "run.log"))))
        log(f"{name} launched {rid}")
    if procs:
        wait_all(procs, STAGE_DEADLINE_S, name)


def run_sweep(width):
    summary_f = os.path.join(ROOT, "summary.json")
    summary = json.load(open(summary_f)) if os.path.exists(summary_f) else {}
    inc = dict(BASE)
    stages = [
        ("S1-lr", "lr", [2e-4, 1e-4, 1e-5]),
        ("S2-optim", "enc_mult", [1.0, 0.3, 0.1, 0.0]),   # per-group encoder lr; 0.0 == frozen encoder (causal)
        ("S3-alpha", "alpha", [0.0, 0.005, 0.01]),
        ("S4-beta", "beta", [0.1, 0.2, 0.05]),
        ("S5-s", "s", [0.9, 0.8, 0.3]),
        ("S6-margin", "margin", [0.0, 0.03]),
    ]   # NN estimator + round-based schedule are PARALLEL BACKUPS (launched in main, not in the main chain)
    for name, key, values in stages:
        cfgs = []
        for v in values:
            c = dict(inc); c[key] = v
            cfgs.append(c)
        log(f"=== {name}: {[run_id(c) for c in cfgs]} (incumbent {run_id(inc)}) ===")
        launch_stage(name, cfgs, width)
        win, tbl = stage_pick([run_id(c) for c in cfgs], run_id(inc))
        for c in cfgs:
            if run_id(c) == win:
                inc = c
                break
        summary[name] = dict(winner=win, table=tbl, incumbent=inc)
        json.dump(summary, open(summary_f, "w"), indent=2)
        wp = tbl.get(win, {})
        goal_append(f"- **[{name} {time.strftime('%m-%d %H:%M')}] winner `{win}`** — "
                    f"val2 {wp.get('val_mean', 0)*100:.0f}% (trend {wp.get('val_trend', 0)*100:+.0f}), "
                    f"cov_cum {wp.get('cov_cum', 0)*100:.1f}%, varσ {wp.get('var_sig', 0):.4f}; "
                    f"alternatives: " + ", ".join(f"`{r}` {t['score']:.3f}" for r, t in
                                                  sorted(tbl.items(), key=lambda x: -x[1]["score"])) + ".")
        log(f"=== {name} winner: {win} ===")
    return inc, summary


# --------------------------------------------------------------------- figures
def wilson_hw(p, n, z=1.0):
    if n <= 0:
        return 0.0
    den = 1 + z * z / n
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den


def overlay_figures(winner_rid):
    runs = {d: load_hist(d) for d in sorted(os.listdir(ROOT))
            if os.path.isdir(os.path.join(ROOT, d)) and load_hist(d)}
    if not runs:
        return
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
    for rid, h in runs.items():
        it = [r["iter"] for r in h]
        win = rid == winner_rid
        kw = dict(lw=2.4, color="#d62728", zorder=5) if win else dict(lw=1.0, alpha=0.55)
        l0, = ax[0].plot(it, [v * 100 for v in series(h, "validity")], label=rid if win else None, **kw)
        if win:
            v = series(h, "validity")
            hw = [wilson_hw(x, N_MEASURE * len(GAMMAS)) * 100 for x in v]
            ax[0].fill_between(it, [x * 100 - w for x, w in zip(v, hw)],
                               [x * 100 + w for x, w in zip(v, hw)], color="#d62728", alpha=0.15)
        ax[1].plot(it, [r.get("var_sigma", 0) for r in h], **kw)
        ax[2].plot(it, [v * 100 for v in series(h, "coverage_cum")], **kw)
        ax[3].plot(it, [v * 100 for v in series(h, "coverage_final")], **kw)
    for a, t in zip(ax, [f"A) validity2 % (γ-mean, winner ±1σ band, n={N_MEASURE}/γ)", "B) var(σ) (candidate spread)",
                         "C) coverage_cumulative %", f"D) coverage_final % (ceiling {N_MEASURE/252*100:.1f}%)"]):
        a.set_xlabel("iteration"); a.set_title(t, fontsize=10); a.grid(alpha=.25)
    ax[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Constructive sweep 0703 — all runs, red = winner `{winner_rid}`", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "sweep0703_overlay.png"), dpi=130); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.4))
    for gi, g in enumerate(GAMMAS):
        for rid, h in runs.items():
            it = [r["iter"] for r in h]
            win = rid == winner_rid
            kw = dict(lw=2.2, color="#d62728", zorder=5) if win else dict(lw=0.9, alpha=0.5)
            axs[gi].plot(it, [r[f"g{g}"]["validity"] * 100 for r in h], **kw)
        axs[gi].set_title(f"validity2 γ={g}"); axs[gi].grid(alpha=.25); axs[gi].set_xlabel("iteration")
    fig.suptitle("Per-γ validity2 (single model, all runs; red = winner) — cross-γ forgetting check", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "sweep0703_validity_perg.png"), dpi=130); plt.close(fig)


def entanglement_figure(winner_rid):
    # compare the winner against the frozen-encoder run (enc_mult=0 == the causal freeze arm)
    frz = [d for d in os.listdir(ROOT) if d.endswith("_e0") and load_hist(d)]
    pairs = [(winner_rid, "#d62728", "winner")] + ([(frz[0], "#1f77b4", "frozen-encoder (enc_mult=0)")] if frz else [])
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.4))
    for rid, col, lab in pairs:
        h = load_hist(rid)
        recs = [r for r in h if "upd" in r]
        it = [r["iter"] for r in recs]
        if recs:
            ctxg = [np.mean([r["upd"].get(f"grad_{k}", 0) for k in ("E_g", "E_l", "GRU")]) for r in recs]
            fldg = [np.mean([r["upd"].get(f"grad_{k}", 0) for k in ("trunk", "head")]) for r in recs]
            ax[0].plot(it, ctxg, "-", color=col, label=f"{lab}: context enc")
            ax[0].plot(it, fldg, "--", color=col, label=f"{lab}: field (trunk+head)")
        it2 = [r["iter"] for r in h]
        ax[1].plot(it2, [r["probes"]["ctx_drift"] for r in h], color=col, label=lab)
        ax[2].plot(it2, [r["probes"]["demo_val_cfm"] for r in h], color=col, label=lab)
    ax[0].set_yscale("log"); ax[0].set_title("per-module grad RMS: context vs field")
    ax[1].set_title("context drift ‖ctx_t−ctx_0‖/‖ctx_0‖ (frozen probes)")
    ax[2].set_title("demo-CFM: distance explored from dataset (↑ expected; not a loss)")
    for a in ax:
        a.grid(alpha=.25); a.legend(fontsize=8); a.set_xlabel("iteration")
    fig.suptitle("Entangled-input-space diagnosis: is the context map chasing the field?", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "sweep0703_entanglement.png"), dpi=130); plt.close(fig)


def launch_backups(width):
    """PARALLEL BACKUP plans (positive-only, at BASE config): NN estimator, round-based fine-tune. Launched
    concurrently with the main chain; each is a clean A/B vs the base GP-online run (same everything else)."""
    procs = []
    for extra in ({"unc": "nn"}, {"schedule": "round"}):
        c = dict(BASE); c.update(extra); rid = run_id(c)
        outdir = os.path.join(ROOT, rid)
        if os.path.exists(os.path.join(outdir, "history.json")):
            log(f"backup {rid} cached")
            continue
        os.makedirs(outdir, exist_ok=True)
        procs.append((rid, *spawn(build_cmd(c, rid, outdir, width), os.path.join(outdir, "run.log"))))
        log(f"BACKUP launched {rid} (parallel to main chain)")
    return procs


def report_backups(summary):
    base_rid = run_id(BASE)
    nn_rid = run_id({**BASE, "unc": "nn"})
    rd_rid = run_id({**BASE, "schedule": "round"})
    _, tbl = stage_pick([base_rid, nn_rid, rd_rid], base_rid)
    summary["backups"] = dict(table=tbl)
    json.dump(summary, open(os.path.join(ROOT, "summary.json"), "w"), indent=2)
    if base_rid in tbl:
        b = tbl[base_rid]
        useful = []
        for rid, lab in ((nn_rid, "NN-estimator"), (rd_rid, "round-based")):
            if rid in tbl and tbl[rid]["score"] > b["score"] + 0.01:
                useful.append(f"{lab} `{rid}` BEATS base (score {tbl[rid]['score']:.3f} vs {b['score']:.3f}, "
                              f"val2 {tbl[rid]['val_mean']*100:.0f}% vs {b['val_mean']*100:.0f}%)")
        verdict = ("USEFUL: " + "; ".join(useful)) if useful else "neither backup beat the GP-online base"
        goal_append(f"- **[BACKUPS {time.strftime('%m-%d %H:%M')}]** (parallel, positive-only at base) — {verdict}. "
                    + "; ".join(f"`{r}` {t['score']:.3f} (val2 {t['val_mean']*100:.0f}%, cov {t['cov_cum']*100:.1f}%)"
                                for r, t in sorted(tbl.items(), key=lambda x: -x[1]["score"])) + ".")
        log(f"backup verdict: {verdict}")


# --------------------------------------------------------------------- main
def main():
    os.makedirs(ROOT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    log("===== run_sweep_0703 master start (POSITIVE-ONLY; GP-online main + NN/round backups) =====")
    width = phase_pretrain()
    backups = launch_backups(width)                     # kick off parallel backups BEFORE the main chain
    inc, summary = run_sweep(width)
    winner = run_id(inc)
    log(f"===== main sweep complete, winner {winner}; waiting on {len(backups)} backups =====")
    if backups:
        wait_all(backups, STAGE_DEADLINE_S, "backups")
    report_backups(summary)

    wdir = os.path.join(ROOT, winner)
    fe_dir = os.path.join(wdir, "final_eval")
    if not os.path.exists(os.path.join(fe_dir, "history.json")):
        os.makedirs(fe_dir, exist_ok=True)
        cmd = [PY, "grid_expand2.py", "--policy", os.path.join(wdir, "final.pt"), "--outdir", fe_dir,
               "--run-id", winner + "-final100", "--iters", "0", "--n-measure", "100",
               "--wandb-mode", "disabled"]
        p, f = spawn(cmd, os.path.join(fe_dir, "run.log"))
        wait_all([(winner + "-final100", p, f)], 2 * 3600, "final-eval")
    fe = load_hist(os.path.join(winner, "final_eval"))
    if fe:
        r0 = fe[0]
        line = " · ".join(f"γ{g}: val2 {r0[f'g{g}']['validity']*100:.0f}% "
                          f"covfin {r0[f'g{g}']['n_final']}/252 reach {r0[f'g{g}']['reach_rate']*100:.0f}%"
                          for g in GAMMAS)
        goal_append(f"- **[FINAL {time.strftime('%m-%d %H:%M')}] winner `{winner}` (W{width}), "
                    f"100 deploys/γ** — {line}.")
        log(f"final eval: {line}")

    overlay_figures(winner)
    entanglement_figure(winner)
    import shutil
    shutil.copyfile(os.path.join(wdir, "final.pt"), os.path.join(ROOT, "best.pt"))

    for mode, out in (("multimodal", "expand2_multimodal_g0.5"), ("progress", f"expand2_progress")):
        cmd = [PY, "grid_expand_viz2.py", "--rundir", wdir, "--mode", mode,
               "--policy0", f"pretrained2_w{width}.pt"]
        p, f = spawn(cmd, os.path.join(ROOT, f"viz_{mode}.log"))
        wait_all([(f"viz-{mode}", p, f)], 3600, "viz")

    h = load_hist(winner)
    covc = series([r for r in h if r["iter"] > 0], "coverage_cum")
    rising = len(covc) >= 3 and (covc[-1] - covc[-3]) > 0.01
    goal_append(f"- **[SWEEP DONE {time.strftime('%m-%d %H:%M')}]** best model `results/sweep_0703/best.pt`; "
                f"figures: sweep0703_overlay / sweep0703_validity_perg / sweep0703_entanglement; videos: "
                f"expand2_multimodal_g0.5 / expand2_progress_{winner}."
                + (" **Caveat: coverage_cum still rising at 2k iters — ranking may shift with longer runs;"
                   " consider rerunning top-2 at 10k.**" if rising else ""))
    log("===== master done =====")


if __name__ == "__main__":
    main()
