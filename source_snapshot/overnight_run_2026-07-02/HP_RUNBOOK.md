# HP_RUNBOOK — distributed fine-tuning of the H_P chessboard SafeFlow expansion (2026-07-05)

**This is the ONE file a remote machine needs.** Everything runs from `overnight_run_2026-07-02/` (self-contained;
no other folder required). Goal: tune the expansion knobs so **coverage (252 staircases) grows while validity2
holds near the pretrained baseline** — report each run with the **tree viz + the 4-panel trend** (many iterations
are now affordable).

## 0. Setup (once per machine)
```bash
cd overnight_run_2026-07-02
# python ≥3.10, torch (CUDA), matplotlib, numpy, wandb (optional: --wandb-mode disabled works everywhere)
# 1) regenerate the dataset (NOT in git; ~25 min CPU, 667 trajs/γ = 207k windows):
python stage2_grid_data.py --seeds 150            # seeds 0-149  → dataset/windows_g{0.1,0.5,1.0}.pt
python gen_more_data.py --s0 150 --s1 600         # +450 trajs/γ (appends)
python gen_more_data.py --s0 600 --s1 667         # +67 trajs/γ  (appends)
# 2) pretrained model IS in git: results/hp_arch/res2w256_ft.pt  (loader: hp_arch_sweep.load_arch)
#    (val-cfm 0.799 · validity2@it0 n=25/γ ≈ γ0.5 72 / γ1.0 92 / γ0.1 64 · reach 100%)
```

## 1. LOCKED BASE (do not change unless the experiment says so)
temp **1.5** explore / **1.0** measure · ell **0.5** (calibrated; 0.2 is DEAD — σ≡1) · enc_lr_mult **0.5** ·
β **0.1** · s **0.9** · N **64** · lr **2e-4** (Adam+cosine) · α **0** · inner **12×128** · traj-level validity2
gate · 252-coverage · GP-RBF λ1e-2, gp_buf 384 · T 250 · n_measure 25/γ · **measure every 100 iters**.
NB: expansion logs print per-γ columns in the order **(0.5, 1.0, 0.1)**.

## 2. Command template (one run = one knob override)
```bash
export CUDA_VISIBLE_DEVICES=<gpu>
python grid_hp_expt.py --iters 2000 --temp 1.5 --ell 0.5 --enc-lr-mult 0.5 --measure-every 100 \
  --arch-ckpt results/hp_arch/res2w256_ft.pt \
  --outdir results/hp_dist/<RUN_NAME> --name hp-<RUN_NAME> --wandb-mode disabled \
  <OVERRIDE FLAGS>
# log: tee or redirect; keep it — the 4-panel plot parses it.
```

## 3. EXPERIMENT MATRIX (one run per row; 2000 iters each, ~80 min/GPU)
**USER VERDICT 2026-07-05: plain-knob arms are CUT** (β/enc/lr separated-effects closed: beta0.01 68→36 ·
enc0 79→35 drift≡0 · lr1e-5 81→21 with γ0.1→0; trees 77→{0,5,15} reached — no plain knob prevents collapse).
**Only the mechanism arms continue: dfrac (2.1) and LwF (2.2).**
**MACHINE SPLIT: REMOTE (ssh, Caltech box) = mechanism brackets · LOCAL (main) = aggressive search**
(defaults dfrac0.25 + lwf0.1, the combined run, winner long-run, quota-D harvest — running locally, do NOT duplicate).
| run name | override flags | tests | machine |
|---|---|---|---|
| dfrac0.1 | `--demo-frac 0.1` | 2.1 replay, light: 10% of every batch = demo windows | **REMOTE** |
| dfrac0.25 | `--demo-frac 0.25` | 2.1 default (32 demo + 96 positives per 128-batch) | local (running) |
| dfrac0.5 | `--demo-frac 0.5` | 2.1 heavy replay (the old v1 anchor regime) | **REMOTE** |
| lwf0.01 | `--lwf-eta 0.01` | 2.2 LwF, light field-distillation on demo ctx | **REMOTE** |
| lwf0.1 | `--lwf-eta 0.1` | 2.2 default | local (running) |
| lwf1.0 | `--lwf-eta 1.0` | 2.2 strong hold | **REMOTE** |
| dfrac0.25+lwf1.0 | `--demo-frac 0.25 --lwf-eta 1.0` | tier 2: replay + strong anchor | **REMOTE** (after singles) |
| dfrac0.5+lwf0.1 | `--demo-frac 0.5 --lwf-eta 0.1` | tier 2: heavy replay + light anchor | **REMOTE** (after singles) |
| dfrac0.25+lwf0.1 | `--demo-frac 0.25 --lwf-eta 0.1` | the combined default | local |

Remote order: dfrac0.1 → lwf1.0 (chain A) · dfrac0.5 → lwf0.01 (chain B); tier-2 combos after the singles
land. 2 GPUs → ~3 h for the singles.

**What 2.1/2.2 are** (implemented in `grid_expand2.py`, inert unless flagged):
- `--demo-frac δ` — every 128-window update batch = δ·128 uniformly-sampled DEMO windows + (1−δ)·128 positives
  (MACE-multihead-finetuning-style replay; v1's demo_frac revived as a tunable).
- `--lwf-eta η` — adds `η · E_{ctx∼demo} ‖v_θ(x_τ,τ|ctx) − v_θ₀(x_τ,τ|ctx)‖²` with v_θ₀ = frozen copy of the
  pretrained field (made at expansion start): holds the field ON OLD INPUTS, leaves new input space free.

## 4. REQUIRED outputs per run (the standing report pair)
```bash
# (a) TREE VIZ — 3 rows: pretrained / it1000 / it2000, at the run's temp/β/ell:
python hp_tree_viz.py --ckpts results/hp_arch/res2w256_ft.pt \
  results/hp_dist/<RUN>/ckpt_1000.pt results/hp_dist/<RUN>/ckpt_2000.pt \
  --labels ft it1000 it2000 --gamma 0.5 --temp 1.5 --ell 0.5 --beta <run beta> --tag <RUN>
# (b) 4-PANEL TREND from the run log:
python hp_trend_viz.py --log <path to the run's log> --out arm_<RUN>_trend.png
# figures land in figures/hp_test/ — send both PNGs + the final 3 log lines.
```

## 5. Judging (fill one row per run)
| run | val2 γ-mean it0→2000 (hold ≥~76?) | jiggle (std of blocks) | cov_cum | SOCP-viol trend | drift | demoCFM | tree: branches/died/reached at it2000 |

Reference results from the local machine (base, 1000-iter arms, FINAL — this is why the plain rows were cut):
plain expansion DEGRADES validity at every single-knob setting — beta0.01: 68→36% (cov 7.4) · enc0: 79→35%
(cov 12.8, **drift≡0.000 all run** → forgetting is in the FIELD/trunk+head, not the encoder) · lr1e-5: 81→**21%**
(cov 13.5, **γ0.1→0** — the WORST arm: tiny steps still walk the same biased direction, 12×128×1000 ≈ 12k
gradient applications). β/enc/lr change the SIZE or LOCUS of the update but not its DIRECTION — the collapse is
a data-composition bias (positives crowd out the pretraining distribution). demo_frac and LwF are the only
knobs that mix old-input gradients back in, and the local mid-run reads confirm it (dfrac0.25 ~57-61% & lwf0.1
~55% at it1200-1500 with cov ~23-24% and demoCFM 0.85-0.90 vs plain arms' 31-39%/1.20 at matched iters).
Winner = holds val2 γ-mean ≥~76 while cov_cum climbs ≥ the ~13%/1k-iter discovery rate.

## 6. Context (1 paragraph)
Model = `GridHPFlowPolicy` res2w256_ft: ctx = raw low5(5) ⊕ E_hp(H_P 1-ch [1,16,12]→CNN+AAP→32); trunk
[20+37+32]→ResMLP(2×256)→head 20; u_max 1. Expansion = σ-tilt candidate sampling (GP-RBF over φ_s, ell 0.5) +
traj-level validity2 gate (approach ∧ taskspace ∧ SOCP) + positive-only cfm updates. Full history/details:
`goal_07_02_Hp.md` + `GOAL_07_02.md` Part 5.
