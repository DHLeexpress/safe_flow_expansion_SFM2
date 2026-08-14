"""Reduced-model experiment driver (user 2026-07-03, 'BEFORE ALL of this').

The reduced model = ctx [relgoal,vel,γ]=5 raw, NO grid CNN / NO E_l / NO GRU — only the velocity field learns,
so the entangled LEARNED context representation is gone. Question: can positive-only safe-flow-expansion reach
the GOAL (coverage+validity) with this minimal setup? Positive-only, GP-RBF online MAIN chain (no enc_mult stage
— there is no encoder), plus NN-estimator and round-based fine-tune PARALLEL BACKUPS. Full-encoded sweep is HELD.
Reuses helpers from run_sweep_0703; writes to results/sweep_0703_reduced/.
"""
import os
import json
import shutil
import time

import _paths  # noqa: F401
import run_sweep_0703 as R

R.ROOT = os.path.join(R.HERE, "results", "sweep_0703_reduced")   # redirect the reused helpers' ROOT
ROOT = R.ROOT
POLICY = "pretrained2_reduced.pt"   # arch flags (use_gru/encode_low/use_grid) live in its checkpoint config
# no enc_mult / freeze in the reduced sweep (there is no learned encoder)
BASE = dict(lr=2e-4, alpha=0.0, beta=0.1, s=0.9, margin=0.0, unc="gp", schedule="online")


def rid_of(c):
    rid = f"lr{c['lr']:g}_a{c['alpha']:g}_b{c['beta']:g}_s{c['s']:g}"
    if c["margin"] > 0:
        rid += f"_m{c['margin']:g}"
    if c.get("unc", "gp") != "gp":
        rid += f"_{c['unc']}"
    if c.get("schedule", "online") != "online":
        rid += f"_{c['schedule']}"
    return rid


def cmd_of(c, rid, outdir):
    cmd = [R.PY, "grid_expand2.py", "--policy", POLICY, "--outdir", outdir, "--run-id", rid, "--iters", "2000",
           "--lr", f"{c['lr']:g}", "--alpha", f"{c['alpha']:g}", "--beta", f"{c['beta']:g}", "--s", f"{c['s']:g}",
           "--n-measure", str(R.N_MEASURE), "--unc", c.get("unc", "gp"),
           "--schedule", c.get("schedule", "online"), "--wandb-mode", "online"]
    if c["margin"] > 0:
        cmd += ["--pos-margin", f"{c['margin']:g}"]
    return cmd


def launch(name, cfgs):
    procs = []
    for c in cfgs:
        rid = rid_of(c); outdir = os.path.join(ROOT, rid)
        if os.path.exists(os.path.join(outdir, "history.json")):
            R.log(f"{name}: {rid} cached"); continue
        os.makedirs(outdir, exist_ok=True)
        procs.append((rid, *R.spawn(cmd_of(c, rid, outdir), os.path.join(outdir, "run.log"))))
        R.log(f"{name} launched {rid}")
    if procs:
        R.wait_all(procs, R.STAGE_DEADLINE_S, name)


def main():
    os.makedirs(ROOT, exist_ok=True); os.makedirs(R.FIG, exist_ok=True)
    R.log("===== REDUCED experiment (ctx=5, no learned encoder; positive-only) =====")
    while not os.path.exists(os.path.join(R.HERE, POLICY)):
        R.log("waiting for pretrained2_reduced.pt ..."); time.sleep(30)

    # parallel backups at base (NN estimator, round-based schedule)
    backups = []
    for extra in ({"unc": "nn"}, {"schedule": "round"}):
        c = dict(BASE); c.update(extra); rid = rid_of(c); outdir = os.path.join(ROOT, rid)
        if not os.path.exists(os.path.join(outdir, "history.json")):
            os.makedirs(outdir, exist_ok=True)
            backups.append((rid, *R.spawn(cmd_of(c, rid, outdir), os.path.join(outdir, "run.log"))))
            R.log(f"BACKUP launched {rid} (parallel)")

    summary = {}
    inc = dict(BASE)
    stages = [("S1-lr", "lr", [2e-4, 1e-4, 1e-5]),
              ("S3-alpha", "alpha", [0.0, 0.005, 0.01]),
              ("S4-beta", "beta", [0.1, 0.2, 0.05]),
              ("S5-s", "s", [0.9, 0.8, 0.3]),
              ("S6-margin", "margin", [0.0, 0.03])]
    for name, key, vals in stages:
        cfgs = [{**inc, key: v} for v in vals]
        R.log(f"=== {name}: {[rid_of(c) for c in cfgs]} (incumbent {rid_of(inc)}) ===")
        launch(name, cfgs)
        win, tbl = R.stage_pick([rid_of(c) for c in cfgs], rid_of(inc))
        for c in cfgs:
            if rid_of(c) == win:
                inc = c; break
        summary[name] = dict(winner=win, table=tbl, incumbent=inc)
        json.dump(summary, open(os.path.join(ROOT, "summary.json"), "w"), indent=2)
        wp = tbl.get(win, {})
        R.goal_append(f"- **[REDUCED {name} {time.strftime('%m-%d %H:%M')}] winner `{win}`** — "
                      f"val2 {wp.get('val_mean', 0)*100:.0f}% (trend {wp.get('val_trend', 0)*100:+.0f}), "
                      f"cov_cum {wp.get('cov_cum', 0)*100:.1f}%, varσ {wp.get('var_sig', 0):.4f}; alts: "
                      + ", ".join(f"`{r}` {t['score']:.3f}" for r, t in sorted(tbl.items(), key=lambda x: -x[1]["score"])) + ".")
        R.log(f"=== {name} winner: {win} ===")
    winner = rid_of(inc)
    R.log(f"===== reduced main chain done, winner {winner}; waiting on backups =====")
    if backups:
        R.wait_all(backups, R.STAGE_DEADLINE_S, "backups")

    # backup verdict (vs base gp-online)
    base_rid = rid_of(BASE); nn_rid = rid_of({**BASE, "unc": "nn"}); rd_rid = rid_of({**BASE, "schedule": "round"})
    _, btbl = R.stage_pick([base_rid, nn_rid, rd_rid], base_rid)
    if base_rid in btbl:
        b = btbl[base_rid]; useful = []
        for rid, lab in ((nn_rid, "NN-estimator"), (rd_rid, "round-based")):
            if rid in btbl and btbl[rid]["score"] > b["score"] + 0.01:
                useful.append(f"{lab} `{rid}` beats base (val2 {btbl[rid]['val_mean']*100:.0f}% vs {b['val_mean']*100:.0f}%)")
        verdict = ("USEFUL: " + "; ".join(useful)) if useful else "neither backup beat GP-online base"
        R.goal_append(f"- **[REDUCED backups {time.strftime('%m-%d %H:%M')}]** — {verdict}. "
                      + "; ".join(f"`{r}` {t['score']:.3f} (val2 {t['val_mean']*100:.0f}%)"
                                  for r, t in sorted(btbl.items(), key=lambda x: -x[1]["score"])) + ".")

    # final eval of the winner (100 deploys/γ)
    wdir = os.path.join(ROOT, winner); fe = os.path.join(wdir, "final_eval")
    if not os.path.exists(os.path.join(fe, "history.json")):
        os.makedirs(fe, exist_ok=True)
        cmd = [R.PY, "grid_expand2.py", "--policy", os.path.join(wdir, "final.pt"), "--outdir", fe,
               "--run-id", winner + "-final100", "--iters", "0", "--n-measure", "100", "--wandb-mode", "disabled"]
        R.wait_all([(winner + "-final100", *R.spawn(cmd, os.path.join(fe, "run.log")))], 2 * 3600, "final-eval")
    feh = R.load_hist(os.path.join(winner, "final_eval"))
    if feh:
        r0 = feh[0]
        line = " · ".join(f"γ{g}: val2 {r0[f'g{g}']['validity']*100:.0f}% covfin {r0[f'g{g}']['n_final']}/252 "
                          f"reach {r0[f'g{g}']['reach_rate']*100:.0f}%" for g in R.GAMMAS)
        R.goal_append(f"- **[REDUCED FINAL {time.strftime('%m-%d %H:%M')}] winner `{winner}` (ctx=5), 100 deploys/γ** — {line}.")
        R.log(f"reduced final: {line}")

    R.overlay_figures(winner); R.entanglement_figure(winner)
    for src, dst in (("sweep0703_overlay.png", "reduced_overlay.png"),
                     ("sweep0703_validity_perg.png", "reduced_validity_perg.png")):
        s = os.path.join(R.FIG, src)
        if os.path.exists(s):
            shutil.copyfile(s, os.path.join(R.FIG, dst))
    shutil.copyfile(os.path.join(wdir, "final.pt"), os.path.join(ROOT, "best.pt"))
    for mode in ("multimodal", "progress"):
        cmd = [R.PY, "grid_expand_viz2.py", "--rundir", wdir, "--mode", mode, "--policy0", POLICY]
        R.wait_all([(f"viz-{mode}", *R.spawn(cmd, os.path.join(ROOT, f"viz_{mode}.log")))], 3600, "viz")
    R.goal_append(f"- **[REDUCED DONE {time.strftime('%m-%d %H:%M')}]** best `results/sweep_0703_reduced/best.pt`; "
                  f"figures reduced_overlay / reduced_validity_perg / sweep0703_entanglement.")
    R.log("===== reduced experiment done =====")


if __name__ == "__main__":
    main()
