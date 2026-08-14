# GOAL_SHORTER — H = 10 → 5 (shorter window/verifier), end-to-end on the EXISTING 0702 files

**Intent (user 2026-07-06):** run the *entire* current chessboard pipeline — data → pretrain → safe expansion →
viz/report — but with the MPPI/FM **window horizon H = 5** instead of 10 (the nominal arc AND the verifier
polytope both certify a 5-step window). **No new parallel files**: a different agent opens the existing scripts
(`stage2_grid_data.py`, `gen_dr_data.py`, `hp_arch_sweep.py`, `grid_hp_expt.py`, `grid_expand2.py`, the viz
scripts) and they work at H=5 after **one knob + two default-unifications**. H is written as ONE constant so
"5, potentially changing" is a one-line edit later.

---
## 0. WORK IN A COPY, then ONE knob (`grid_feats.H_PRED`) drives FOUR things
**Isolation (Q2 = copy):** the H=10 runs stay live, so H=5 goes in a fresh sibling folder:
```bash
cp -r overnight_run_2026-07-02 overnight_run_H5 && cd overnight_run_H5
rm -rf results dataset wandb figures                 # start clean; keep the .py + .md
```
`_paths.py` recomputes relative to the new folder, so the SHARED deps still resolve
(`overnight_run_2026-07-01/verifier_polytope.py` + `di_grid_viz`, `overnight_run_today/src/*`) — the H5 folder
must remain a sibling of `overnight_run_2026-07-01/` under the repo root. **Local files** (copied, editable):
`grid_feats.py`, `grid_scene.py`, `grid_metrics.py`, `grid_metrics2.py`, all `hp_*`/`grid_*`. **Shared, DO NOT
EDIT:** `overnight_run_2026-07-01/verifier_polytope.py` — it's used by H=10 too; we drive it via an explicit
`H_win` argument instead (Edit C).

**Q1 = expert also plans H=5** (user: same sensing range ⇒ 5-level-set sampling differs, a slice of a 10-plan is
NOT the same controls). So `GF.H_PRED` drives FOUR things: MPPI expert horizon, FM window, validity2 window,
verifier window. Today only model/data/sampling follow it; the other three are hardcoded `10`. The edits:

**Edit A — the knob** · `grid_feats.py:25` → `H_PRED = 5`
**Edit E — MPPI expert horizon** · `grid_scene.py` `mode1_config()`: after `cfg = dict(load_best_config())`
add `cfg["horizon"] = GF.H_PRED` and `cfg["guidance_horizon"] = GF.H_PRED` (add `import grid_feats as GF` if
absent). This makes the SafeMPPI expert genuinely plan 5 steps (was `horizon:10` in best_area_mode4.json;
guidance_horizon 12 would exceed 5). ⇒ different, shorter-sighted demos — intended.
**Edit B — validity2 window** · `grid_metrics2.py`: DEFAULTS `traj_valid2(..., H=10, ...)` and
`traj_breakdown(..., H=10, ...)` → `H=GF.H_PRED` (module already imports GF). All callers use the default.
**Edit C — SOCP verifier window** · `grid_metrics.py:89` `socp_ok(..., H_win=10)` → `H_win=GF.H_PRED` (add
`import grid_feats as GF`). `socp_ok` ALREADY forwards `H_win` explicitly to `certify_trajectory` (line 92-93),
and `certify_window`'s `alpha=(1-γ)**arange(seg_len)` follows the passed seg — so the SHARED
`verifier_polytope.py` needs **NO edit** and H=10 stays intact. (Q3: H=5 window = 5 controls, 6 states; robot's
first state has H_P≡1 by construction ⇒ **5 shrinking level sets** α^1…α^5 — matches "5 level sets".)
**Edit D — tree-viz schedule** · `hp_tree_viz.py`: branch schedule is H-tied. At H=5 a node is one 5-step
window (0.5 s at dt 0.1); use `k = [6,6,5,5,4,4,3,3,2,2,1,1,1,…]` extended to ~T/H = 50 nodes (pad 1s).

**What cascades AUTOMATICALLY (verify, do NOT edit):**
- **Model dim** — `GridHPFlowPolicy(H_pred=GF.H_PRED)` default ⇒ d = 2H = **10**, trunk in **79** (U10+ctx37+t32),
  head `Linear(256→10)`, φ_s stays 256. `hp_arch_sweep.build("res2w256")` unchanged, ~same params minus 10 I/O.
- **Data slicing** — `stage2_grid_data.windows_from(H=GF.H_PRED)`; `gen_dr_data` reuses `windows_from` ⇒ H=5 windows.
- **Sampling noise templates** — `grid_rollout.py:36,45` and `grid_policy2.py:203` use `GF.H_PRED`.
- **`certify_window`** alpha length = `traj_c.shape[0]` = passed seg length ⇒ follows H_win.

**⚠ Comparability / isolation:** changing `H_PRED` globally makes ALL H=10 checkpoints/datasets unloadable
(d mismatch) and shifts validity2 baselines (shorter arcs pass more easily) — H=5 numbers are NOT comparable to
any H=10 result. Do H=5 **after** the current H=10 runs + their post-hoc viz finish, OR copy the 0702 folder to
a fresh `overnight_run_H5/` and edit there (keeps H=10 artifacts intact). See open question Q2.

---
## 1. DATA (4002 trajs = 1334/γ: 2001 origin + 2001 off-diagonal), sliced at H=5
Scene/expert IDENTICAL to now (env built once, only start varies). Backup H=10 shards first.
```bash
mkdir -p dataset/backup_H10 && cp dataset/windows_g*.pt dataset/backup_H10/   # if editing in place
python stage2_grid_data.py --seeds 667                         # origin starts, 667/γ  (fresh windows_g*.pt, H=5)
python gen_dr_data.py --seeds 667 --offdiag 0.5 --out-prefix "" --append       # off-diag |y-x|≥0.5, +667/γ
python od_viz.py                                               # sanity viz of off-diag experts (figures/dr_test_overnight/)
```
Expect ~1334/γ successes; window count ≈ half the H=10 count (shorter slices → more windows per traj, net similar).

## 2. PRETRAIN the FM (dim changes, recipe stays) → `res2w256_ft_v2` at H=5
Backbone unchanged except d 20→10 (user spec). Train res2w256 on the merged 4002 (build auto-sizes at H=5):
```bash
python -c "import _paths, hp_arch_sweep as A; p=A.build('res2w256'); p,bv=A.train(p,'res2w256_ft_v2',epochs=120,lr=3e-4); A.evaluate(p,'res2w256_ft_v2'); print('val-cfm',bv)"
```
(Or reuse `hp_finetune_v2.py` if you first build a base — but from-scratch 120 ep on 4002 is the clean H=5 base.)
Saves `results/hp_arch/res2w256_ft_v2.pt` (variant-tagged, load_arch-compatible). **Report** val-cfm +
validity2@it0 **n=50** per-γ (see §5) — this is the new H=5 baseline (H=10 was 0.810 / 77% [76/78/76]; expect
DIFFERENT, shorter arcs pass more easily so validity likely ↑; that is fine, just re-baselined).
**Pretrain viz (the moving verifier polytope, now 5 level sets):**
```bash
python grid_verifier_viz.py --ckpt results/hp_arch/res2w256_ft_v2.pt      # verifier_movie(): polytope+DTCBF along FM rollout, per γ
python hp_origin_overlay.py --ckpt results/hp_arch/res2w256_ft_v2.pt --tag v2_H5   # origin rollouts, green=valid2/red=fail
```

## 3. ell RE-CALIBRATION (φ_s now lives in 10-dim control space)
```bash
python hp_s_calib.py --ckpt results/hp_arch/res2w256_ft_v2.pt   # s* (spread); and re-run ell:
python hp_ell_calib.py --temp 1.5                               # ell* for H=5 (σ-std argmax); update the --ell you pass below
```

## 4. SAFE EXPANSION — the `ov_mine` recipe, from the H=5 base, encoder frozen
Locked recipe (winner on H=10): **δ 0.4 · η 0.05 · β 0.1 · α 0.02 · s 0.9 · lr 1e-4 · temp 1.5 · enc_lr_mult 0
(freeze) · grad-clip 10 · ell = ell\*(H5)**. 5k first (extend to 20k if it holds), ckpt every 1k, measure/500 n=50:
```bash
python grid_hp_expt.py --iters 5000 --measure-every 500 --n-measure 50 --ckpt-every 1000 \
  --temp 1.5 --ell <ell*> --enc-lr-mult 0 --lr 1e-4 --grad-clip 10 \
  --demo-frac 0.4 --lwf-eta 0.05 --beta 0.1 --alpha 0.02 --s 0.9 \
  --arch-ckpt results/hp_arch/res2w256_ft_v2.pt \
  --outdir results/hp_H5/ov_mine --name hp-H5-ov-mine --wandb-mode disabled
```
Hypothesis (recall geometric-reset finding): **shorter H recombines sub-trajectories → coverage may go UP** vs
long-H mode-collapse. That is the whole point of the H=5 study — watch cov_cum vs the H=10 ov_mine (31.5%@5k).

## 5. REPORTING (the standing format — user add 2026-07-06: coverage/validity + tree + 4-panel "4×4")
Every run reports the SAME pair, at H=5:
**(a) Coverage + validity, per 500 iters, n=50, per-γ** — read straight from the log lines
`it0XXXX: val2 NN% (γ:g0.5/g1.0/g0.1) cov_cum NN% cov_fin NN% … drift … demoCFM … loss …`.
Primary metric = **val2 γ-mean held ≥ its own it0 anchor** while **cov_cum climbs**; γ0.1 is the tie-break.
**(b) 4-PANEL trend ("4×4")** — `hp_trend_viz.py` (val2 per γ + jiggle · coverage · SOCP-viol · drift+demoCFM):
```bash
python hp_trend_viz.py --log results/hp_H5/ov_mine.log --out dr_test_H5/ov_mine_trend.png --title "H5 ov_mine 5k"
```
**(c) TREE VIZ — 6 trees, one per 1k, legible 2×3 grid** (v2-base + it1000…it5000), at the arm's temp/β, H=5 schedule:
```bash
python hp_tree_viz.py --ckpts results/hp_arch/res2w256_ft_v2.pt \
  results/hp_H5/ov_mine/ckpt_1000.pt … ckpt_5000.pt --labels v2 it1000 … it5000 \
  --gamma 0.5 --temp 1.5 --beta 0.1 --ell <ell*> --ncols 3 --tag H5_ov_mine --outdir figures/dr_test_H5
```
Figures → `figures/dr_test_H5/`. Report table columns: run · val2 it0→peak→final · per-γ · cov@final · demoCFM ·
tree branches/died/reached (v2→it5000) · verdict.

---
## 6. FILE INDEX (what each existing file does — all H=5-ready after §0)
| file | role at H=5 |
|---|---|
| `grid_feats.py` | **H_PRED knob (Edit A)** · axis_grid/low5/hist_pad (unchanged) |
| `grid_metrics2.py` | validity2 gate (**Edit B**) · traj_valid2 / approach / taskspace |
| `grid_metrics.py` + `overnight_run_2026-07-01/verifier_polytope.py` | SOCP verifier / DTCBF polytope (**Edit C**) |
| `stage2_grid_data.py` · `gen_dr_data.py` | data-gen (origin / off-diag), auto H=5 via windows_from |
| `hp_arch_sweep.py` · `grid_policy2.py` · `grid_hp_expt.py` | model build/train, auto d=2H=10 |
| `grid_expand2.py` | safe expansion loop (σ-tilt · validity2 gate · demo_frac/LwF/α/grad_clip) |
| `grid_rollout.py` | receding-horizon `fm_deploy`/`deploy_many` (tracks hist) |
| `grid_verifier_viz.py` | **the moving verifier-polytope animation** (5 level sets) |
| `hp_rollout_viz.py` · `hp_origin_overlay.py` | per-γ / origin rollout overlays |
| `hp_tree_viz.py` | recursive safe-expansion tree (**Edit D** schedule) · `--ncols` grid |
| `hp_trend_viz.py` | 4-panel trend |
| `hp_ell_calib.py` · `hp_s_calib.py` | σ-kernel ell/s recalibration in 10-dim φ_s |

## 7. DECISIONS LOCKED (user 2026-07-06)
- **Q1 → expert plans H=5 too** (Edit E). Same sensing range ⇒ the 5-level-set sample differs from a slice of a
  10-step plan, so the demos must come from a genuine 5-step planner. Data distribution changes vs H=10 — intended.
- **Q2 → copy to `overnight_run_H5/`** (§0). H=10 runs/artifacts untouched; H=5 runs on the spare GPU headroom now.
- **Q3 → H=5 window** (5 controls, 6 states). Robot's first state is H_P≡1 by construction, so it's not counted:
  **5 shrinking level sets** α^1…α^5. Verifier `certify_window` handles this via the passed seg length.

## 8. ONE-SHOT SANITY (before the full regen — proves the knob works end to end)
After Edits A-E, from `overnight_run_H5/`:
```bash
python -c "import _paths, grid_feats as GF, grid_scene as GS, hp_arch_sweep as A; \
print('H_PRED', GF.H_PRED, '| expert horizon', GS.mode1_config()['horizon']); \
p=A.build('res2w256'); print('model d', p.d, '(=2H, expect 10) | trunk_in', p.d+p.ctx_dim+p.t_dim, '(expect 79)')"
python stage2_grid_data.py --seeds 4 --gammas 0.5   # expect U shape [.,5,2]; check windows_g0.5.pt grid/U dims
```
All three must read 5 / 5 / 10 / 79 and the smoke shard must have U of width-5 windows — then run §1-§4.
