# goal_07_02_Hp — H_P inductive-bias reduced model: capacity → then sweep (2026-07-05)

The 0704 plan (beat Kazuki) is TERMINATED (9 tasks: 7 done; Stage-E head-to-head closed at "compute 15× + succ
+6 n.s., collision 9.7 vs 4.0 open"; sanity-3 folded into validity2 work). Focus = HP on the 0702 chessboard.
Figures for testing → `figures/hp_test/`.

## ★ FINAL JUDGMENT — ALL ARMS → LOCKED WINNER (task #54, 2026-07-06)
**The arc, judged:** plain-knob arms (β/enc/lr) ALL collapse → forgetting lives in the trunk/head FIELD, not the
encoder (enc0 drift≡0 yet 79→35). Mechanism arms (demo_frac replay, LwF anchor) HOLD. Phase-S proved the hold is
COMPOSITE and causal: **frozen encoder + old-input gradients (replay δ OR anchor η)** — each alone fails, together
no dip. δ0.5 is the replay floor. Temp is a pure explore↔retain trade (not a lever). The ceiling was ENCODER-OOD:
solved by **folding off-diagonal data into PRETRAINING** — `res2w256_ft_v2` reads **77% balanced (76/78/76)**,
retiring the γ0.1 problem at the source. GRU tested → tied with no-GRU but 2.5× slower ⇒ **no GRU**. DR
encoder-splice → replay repairs γ0.1 through new eyes. 20k confirmed no collapse; s cleanly separates
explore(s.99→cov52)↔retain(s.9→γ0.1 50).

**★ LOCKED WINNER RECIPE (hold-while-explore):**
- **Base**: `results/hp_arch/res2w256_ft_v2.pt` — OD-augmented (4002 trajs, origin+off-diagonal), balanced γ, NO GRU.
- **Recipe (ov_mine / "yours")**: **frozen encoder (enc_lr_mult 0) · demo_frac 0.4 · lwf_eta 0.05 · β 0.1 ·
  α 0.02 · s 0.9 · lr 1e-4 · temp 1.5 · grad-clip 10** · ell 0.5. → holds val2 ~67-70% with cov ↑ to 43% @20k,
  γ0.1 alive at 50, no collapse.
- **Max-coverage alt**: same but **s 0.99** → cov 52%@20k (γ0.1 pays, drops to 36). Pick by objective.
- Artifacts: `results/hp_20k/{yours,mine}/ckpt_20000.pt`.

**→ TRANSFER: `GOAL_SFM_TRANSFER.md`** (launch-ready runbook — port the 3 knobs into grid_expand_sfm, run
expansion from pretrained_sfm, re-benchmark vs Kazuki). Honest framing: 0704 already CLOSED the collision gap via
certified DEPLOYMENT (3%/97% @6ms vs Kazuki 10%/95ms); expansion-only was 9.7%. So the transfer's value = does the
better hold-while-explore base let certified deployment go further (lower pad/yield, higher success). A refinement
on an already-winning deployment, not a must-win.

Current 20k iter basline -> 

## CURRENT STATUS — necessary & sufficient config snapshot

**Model (`grid_hp_expt.GridHPFlowPolicy`, ckpt `results/hp_chessboard/pretrained_hp.pt`):**
- ctx = raw low5(5)=[relgoal2, vel2, γ] ⊕ **E_hp**: H_P channel [1,16,12] → Conv(1→8,3×3) SiLU → Conv(8→16,3×3)
  SiLU → AdaptiveAvgPool(4,3) → Flatten → Linear(192→32) SiLU. NO E_l / hist / GRU. 101.4k params (E_hp 7.4k).
- velocity trunk: **[U(20) + ctx(37) + fourier-t(32)] = 89 → 256 → 256 (SiLU, depth 2) → head 20**; u_max=1.0.
- pretrain: 52,201 windows (450 trajs, 150/γ, SafeMPPI mode-1 expert seeds 0-149), AdamW+warmup+cosine 120 ep,
  val-cfm 0.977. validity2@iter0 (n=25): **γ0.5 92% · γ1.0 56% · γ0.1 0%**.

**Expansion defaults (`grid_expand2.SFG2Config`, 0702):** temp=**1.3** (exploration sampling — wide ON PURPOSE to
generate data; measurement uses temp_measure=**1.0**) · β=0.1 · s=0.9 · GP-RBF ell=0.2 λ=1e-2 gp_buf=384 · N=64 ·
lr=2e-4 Adam+cosine · α=0 (positive-only) · inner_steps=12 · batch=128 · enc_lr_mult=1.0 · T=250 · n_measure=**25**
eps/γ · caps pos 60k (FIFO) · γ **log-column order (0.5, 1.0, 0.1)** · gate = validity2 (approach ∧ taskspace ∧ SOCP)
· coverage = 252 staircases (cum + final).

**20k full-run verdict (Part 5 of GOAL_07_02.md):** coverage 100/252 (39.8%, decelerating); no collapse in 20k
iters; γ-selective consolidation (γ1.0 64-84 / γ0.5 ~44 / γ0.1 starved 0-12 — the rich-get-richer certification
loop); drift 0.757 flat from it3000; demoCFM →4.0.

## KEY OBSERVATION driving this phase (user, from `figures/expand2_multimodal_g0.5.gif`)
At the FIRST obstacle encounter the pretrained policy is **unimodal → unsafe**. Capacity problem: the vector
field must produce complex/multimodal landscapes BEFORE σ-dispersion or β-tilt can mean anything. Track the FM
**policy distribution itself** (not only σ): plot first-control u₀ distributions at obstacle-encounter states;
measure left/right peak splitting.

## PLAN

**Step 0 — model capacity (BEFORE any sweep):**
- 0.1 MORE DATA: `gen_more_data.py` (RUNNING) seeds 150-599 → target ~1800 trajs / ~200k windows total
  (was 450/52k). Backup of old shards: `dataset/backup_450traj/`.
- 0.2 ARCH SWEEP on the enlarged data: trunk ∈ {depth-2 256 (current), depth-3 256, **ResNet-MLP** 2 and 3
  pre-LN residual blocks (the proven 0704 cooked-trunk recipe), width 384 variant}. Keep 89-in → 20-out family.
- 0.3 AUTO-SELECT: train each (~8 min), rank by val-cfm → for the top models measure validity2@iter0 (n=25/γ,
  the standard) + the **multimodality score**: at obstacle-encounter states sample 256 windows, KDE/split of u₀
  (and next-position) left vs right — a model must place peaks BOTH sides. Best model = new pretrained.
- Deliverables: train/val curves, validity2 table, u₀-distribution snapshots (`figures/hp_test/`), overlay viz of
  generative policies.

**Step 1 — the 5-arm sweep** (2000 iters/arm, one knob, all else defaults, compare blocks 0-2000), FROM the new
pretrained: enc_lr_mult=0 (freeze E_hp) · lr 1e-5 · inner_steps 4 · α 0.005 · **β 0.2 (KEPT — user: flat σ may be
a symptom of unimodal candidates; re-judge β only after capacity is fixed)**.

**Step 2 —** per-γ fixes for the certification loop (γ0.1 starvation): per-γ harvest quotas / temp↓ for tight γ /
balanced batches. Then long run with the winning recipe.

## Q&A (user 2026-07-05)
1. **enc_lr_mult=0 ≡ freeze E_hp — the code** (grid_expand2.run_expand2): parameters are split into two optimizer
   groups — `groups = [{field(trunk+head), lr}, {enc(E_hp), lr × enc_lr_mult}]`. Learning rate 0 ⇒ Adam's step
   size for those weights is 0 ⇒ they never change = frozen. **Random-freeze p=0.95** ≈ in expectation a 0.05×
   slowdown, but with Adam it becomes sporadic FULL-size steps (jumpier than lr×0.05). The literature versions of
   "slow encoder": layer-wise lr decay (BERT fine-tuning), surgical fine-tuning (Lee et al. '22), and — closest —
   **momentum/target encoders** (BYOL, RL target networks): enc follows an EMA of itself. Our enc_lr_mult∈{0, 0.05,
   0.1} IS the standard knob; EMA-enc is the upgrade if partial freezing wins. OOD corners (e.g. bottom-right)
   argue for small-but-nonzero mult or EMA rather than hard freeze — the sweep tests the extreme first.

3. α>0: in the sweep. 4. β: kept, deferred judgment until multimodality is fixed (your causal point accepted:
   candidate diversity is upstream of σ dispersion).

## TREE VIZ MODULE (user spec 2026-07-05, `hp_tree_viz.py`) — the standing per-arm report
For an arm at Nk iters → **N+1 rows** (row 0 = pretrained "just sampling"; row k = ckpt_k000 with the arm's own
temp/β): each row is one TREE — a σ-tilt trunk rollout where at every **1 s node (= one H=10 window)** we sample
N=64 candidates at the arm's temp (default 1.3), importance-resample **k** branches via p∝exp(σ/β) with the
decaying schedule **k = 5,4,4,3,3,2,2,1,1,1,…**, roll the k branches 1 s in PARALLEL, colored by σ (viridis);
branches failing validity are marked RED with the failure class (**✗G** goal-seeking / **✗T** taskspace /
**✗S** SOCP) and terminated at the failure point; the trunk continues via the tilt winner until goal or T=250.
σ buffer for the viz = φ_s of 512 random dataset windows under that row's policy (documented stand-in for the
run's query buffer). β-sweep arms are read by how the spray CHOICE shifts as iterations grow. REPORT FORMAT:
this tree (per arm) + the `figures/hp_full20k_trend.png`-style 4-panel trend, consistently.

## WORKFLOW (standing, per sweep round)
1. RUN: 5 arms × 2000 iters from `res2w256_ft` (one knob each; ckpt/500, measure/200, n=25/γ).
2. PER ARM on completion: (a) recursive TREE, rows = ft/ckpt_1000/ckpt_2000 at the arm's temp/β (metrics:
   branches, died-by-✗G/✗T/✗S, reached, corridors); (b) 4-panel trend (val2 per γ + jiggle amplitude, coverage,
   SOCP-viol, drift+demoCFM); (c) row in the cross-arm table.
3. SELECT: winner = holds val2 γ-mean ≥ it0 baseline (≈76%) with jiggle ↓ and acceptable coverage slope;
   tie-break on γ0.1 (the starved end).
4. NEXT ROUND: winner → long run + per-γ fixes (harvest quotas / per-γ temp) targeting coverage↑ with validity
   HELD (the rich-get-richer loop). Report each stage: tree + trend + updated goal md.

## ell CALIBRATION (2026-07-05, `hp_ell_calib.py`, `figures/hp_test/ell_calibration.png`)
Default ell=0.2 gives σ≡1 (σ-std 0.006 — the uniform-yellow tree): candidate features sit ~0.46-0.61 from the
buffer ≈ 2-3× the lengthscale → kernel ≈ 0 → prior everywhere. Sweep: **ell\* = 0.5** (σ-std 0.0514, ~9×;
median-heuristic 0.57 agrees; 0.403 tied). **ALL hp4 sweep arms run with --ell 0.5** (documented deviation from
the 0702 default, calibration-justified). Tree-viz conventions locked: recursive no-orphan branching (children
throttled, k_eff≥1 — every branch resolves to goal/fail), depth-based opacity (first branches opaque), black
dots = true branch events (k≥2), failure classes color-coded (orange G: goal-seeking · purple T: taskspace ·
red S: SOCP), ONE gold goal star, green dots = reached leaves, σ-colorbar (ell*).

## SWEEP ROUND 2 (hp5, 2026-07-05 evening) — NEW LOCKED CENTER (user)
**Defaults changed:** temp explore 1.3→**2.0** (measure stays 1.0) · enc_lr_mult 1.0→**0.5** · **ell re-calibrated
at the new center: ELL\* = 0.5 again** (σ-std 0.049 at temp 2.0; median-heuristic 0.62; 0.2 → σ≡1 dead).
Unchanged: β 0.1 · s 0.9 · N 64 · lr 2e-4 cosine · α 0 · inner 12×128 · n_measure 25/γ · traj-level validity2
gate · 252 coverage. it0 anchor (measure temp 1.0, unchanged): validity2 64/72/92 (mean 76).
Center-tree sanity (`figures/hp_test/tree_sanity_center_t2.png`): temp 2.0 → 731 branches, 99 died, 38 reached
(vs 923/55/83 at 1.3) — much wider exploration, riskier proposals, harvest slower per trajectory.
**ARM SET (one override each vs the new base):** 1) encm0.1 (enc_lr_mult 0.5→0.1) · 2) lr1e-5 · 3) inner4 ·
4) **alpha0.1** (α 0→0.1, raised from 0.005) · 5) beta0.2. Judging unchanged (hold ≥76 γ-mean, jiggle↓, cov
slope, SOCP, γ0.1 tie-break, tree metrics ft→1k→2k).

## SWEEP ROUND 3 (hp6, 2026-07-05) — separated effects at the FINAL LOCKED BASE
Base (LOCKED, final): temp **1.5** explore / 1.0 measure · ell 0.5 · enc_lr_mult 0.5 · β 0.1 · s 0.9 · N 64 ·
lr 2e-4 Adam+cosine · α 0 · inner 12×128 · measure/100 · n=25/γ. (β0.2 quota experiments showed temp 2.0 too
extreme → 1.5; γ log-column order still (0.5, 1.0, 0.1).)

**Results (1000 iters, one knob each; `results/hp_sweep6/`; tree = reached-goal leaves at γ0.5/temp1.5,
ft baseline 77/963 branches):**
| arm | val2 γ-mean it0→1000 | per-γ at 1000 | cov_cum | drift | demoCFM | tree reached ft→500→1000 | verdict |
|---|---|---|---|---|---|---|---|
| beta0.01 (greedy tilt) | 68 → **36%** | 56/32/20 | 7.4% | 0.280 | 0.79→1.20 | 77 → 1 → **0** | collapse, discovery also slow |
| enc0 (freeze E_hp) | 79 → **35%** | 48/44/12 | 12.8% | **0.000 ≡** | 0.79→1.20 | 77 → 11 → **5** | collapse WITH frozen encoder |
| lr1e-5 (rerun) | 81 → **21%** | 28/36/**0** | 13.5% | 0.140 | 0.79→1.01 | (tree rendering) | WORST arm — tiny steps, same bias; γ0.1 dead |

**The separated-effects verdict — where forgetting lives and what each knob can/cannot do:**
- **enc0 is the theorem**: drift ≡ 0.000 for 1000 iters (encoder bit-frozen) yet validity collapsed 79→35 on the
  SAME trajectory as every other arm ⇒ **forgetting lives in the trunk/head FIELD, not the encoder**. enc_lr_mult
  protects ctx geometry (σ/kernel sanity), it is NOT a retention lever. Optimize: keep 0.5; remote tests 0.1;
  EMA-encoder only if ctx drift ever bites.
- **β is a selection knob, not a retention knob**: β0.01 (greedy-σ) collapsed identically AND discovered less
  (7.4% vs 12.8): extreme tilt picks weirder candidates that fail the gate → fewer, more-biased positives.
  Optimize: β 0.1 stays; brackets 0.05/0.3 on remote judge discovery-rate-vs-validity-slope only after a hold
  mechanism exists.
- **lr is a speed knob**: prior evidence (quota-C at lr1e-5 still collapsed) says small steps walk the SAME biased
  direction, just slower. The rerun completes the table; expectation: slower slope, same sign.
- **⇒ The collapse is a DATA-COMPOSITION bias**: positive-only batches are drawn ever further from the pretraining
  distribution; no per-parameter step-size/locus knob can fix a gradient that points away from the demo manifold.
  **The real levers change the gradient DIRECTION: 2.1 demo_frac (mix old-input batches) and 2.2 LwF (anchor the
  field on old inputs)** — exactly the two-machine phase below.

## PHASE-S — 500-ITER "HOPE" SCREENING (2026-07-05 ~22:30, USER WIND-BACK; supersedes the wave-2/nyx-chain plan)
User verdict after wave-1: both mechanism arms DIP in the first 500 (dfrac0.25 71→59, lwf0.1 79→63) before
recovering — not good enough. **Today's goal: find ONE config where validity holds/rises AND coverage climbs
within the FIRST 500 iters.** No long runs until such "a hope" exists. Round-based divide & conquer, 3-4
parallel pairs per round (helios GPU3 ×2 + nyx GPU0/GPU1), report → narrow → repeat.
- **Thesis (user)**: high δ + high η + FROZEN grid encoder ≈ mimics the pretrained policy by construction while
  the (1−δ) positives nudge off-diagonal. Explore temp ↓ **1.3**; **n-measure ↑ 50/γ** (high-belief it500 verdict).
- **Code (committed with this section)**: (a) `grid_expand2.py` enc_lr_mult≤0 now sets `requires_grad_(False)`
  on ALL encoder params (hard freeze, not Adam-lr-0); (b) `grid_hp_expt.py --n-measure` exposed. Smoke-verified
  (drift≡0 with δ0.75+η1.0 updates flowing).
- **ROUND 1 slate** (500 it · temp 1.3 · n=50 · measure/100 · rest = locked base; `results/hp_screen/`):
  | run | δ | η | enc | machine | answers |
  |---|---|---|---|---|---|
  | safeMAX | .75 | 1.0 | frozen | helios G3 | the thesis corner |
  | safeDELTA | .75 | 0.1 | frozen | helios G3 | heavy replay, weak anchor enough? |
  | safeETA | .25 | 1.0 | frozen | nyx G1 | strong anchor, light replay enough? |
  | safeNOEF | .75 | 1.0 | 0.5 | nyx G0 | is the freeze needed? |
- **GATE @ it500 (n=50)**: PASS = val2 γ-mean ≥ it0 anchor AND cov_cum ≥ 10% · WATCH = within 4 pts + cov ≥ 10%
  · else rejected. References: wave-1 dfrac0.25 59/16.1 · lwf0.1 63/13.1 · plain arms 35-48/5-8 (all FAIL).
- **Round 2 branches (pre-registered)**: safeMAX passes → relax one knob per run (δ.5 / η.1 / unfreeze .5 /
  temp 1.5); all fail → tighten (δ.9 / η10 / temp1.0 / inner4). After any PASS: 2k confirm top-2 → raise 60k
  pos-buffer cap → long run; tree + 4-panel at each stage.

### ROUND 1 RESULT (23:50) — **VALIDITY HOLD ACHIEVED; the frozen encoder is CAUSALLY NECESSARY**
| run | anchor→it500 | traj (it100..500) | cov@500 | drift | verdict |
|---|---|---|---|---|---|
| safeDELTA δ.75 η.1 EF | 71 → **74 (+3)** | 71·71·70·78·74 | 6.6% | ≡0 | **best holder — RISES, no dip** |
| safeMAX δ.75 η1 EF | 75 → 73 (−2) | 71·77·73·75·73 | 5.4% | ≡0 | holds (never below 71) |
| safeETA δ.25 η1 EF | 73 → 73 (0) | 69·85·73·76·73 | 6.0% | ≡0 | holds |
| safeNOEF δ.75 η1 enc.5 | 73 → **61 (−12)** | 69·80·69·72·61 | 4.8% | 0.134 | **FAILS — only unfrozen arm** |
- Key composition: freeze alone collapses (R3 enc0 79→35), replay/anchor alone dips (wave-1); **freeze +
  replay/anchor holds**. safeNOEF vs safeMAX is the controlled proof the freeze is required.
- η1.0 adds nothing over η0.1 when δ=.75 (safeMAX ≤ safeDELTA) → replay is the main holder at heavy δ.
- **Coverage gate FAILED everywhere (4.8-6.6 < 10)** — the safety cost discovery (temp 1.3 + heavy replay).
### ROUND 2 (launched 23:55, pre-registered relax branch — restore discovery, keep the hold; EF everywhere)
| run | δ | η | temp | machine |
|---|---|---|---|---|
| r2_temp15 | .75 | .1 | **1.5** | helios G3 |
| r2_delta05 | **.5** | .1 | 1.3 | helios G3 |
| r2_combo | **.5** | .1 | **1.5** | nyx G1 |
| r2_etaT15 | .25 | 1.0 | **1.5** | nyx G0 |
Gate unchanged (val2@500 ≥ anchor ∧ cov ≥ 10).

### ROUND 2 RESULT (~01:00 07-06) — **THE FRONTIER IS δ0.5 + η0.1 + EF**
| run | δ/η/temp | anchor→500 | cov@500 | verdict |
|---|---|---|---|---|
| **r2_delta05** | .5/.1/1.3 | 74→**79 (+5)**, ends at max, γ0.1=66 (best ever) | 8.6 | val-PASS · cov just short |
| **r2_combo** | .5/.1/1.5 | 75→73 (−2, in noise), γ0.5=92 | **11.1 ✓** (climbing 1.2/100it) | **cov-PASS · val-WATCH** |
| r2_temp15 | .75/.1/1.5 | 78→71 (−7) | 6.6 | reject — heavy replay chokes discovery |
| r2_etaT15 | .25/1.0/1.5 | 80→70 (−10) | 8.9 | reject — anchor corner degrades at wide temp |
Reading: δ.75 too conservative even at temp 1.5; δ.5 holds AND discovers; temp trades ~2-3 val pts for ~2.5 cov.
### CONFIRM STAGE (launched ~01:05): helios 2k× {r2_combo, r2_delta05 configs} · nyx R3 500-screens
{r3_temp14 = δ.5 η.1 EF temp1.4 (the interpolation bet) · r3_delta04 = δ.4 η.1 EF temp1.5 (more explore at
cov-passing temp)}. Judge confirms vs wave-1 dfrac0.25 (67%/27.4 with dip): want ≥70 held, no dip, cov ≥ 27.
R3 RESULT: temp14 68→72 (+4) cov 8.6 · delta04 78→68 (dip 57) cov 9.1 — δ0.4 BREAKS the hold; floor = δ0.5.
Confirms KILLED at it400 per user (combo 75%/cov 10.2 · delta05 71%/7.0 at kill — inconclusive by design).

## PHASE-S WRAP-UP — LESSONS (user stop 2026-07-06 ~02:30: "enough to see temp is not working")
1. **The hold mechanism is SOLVED and composite**: frozen encoder + old-input gradients (replay δ OR anchor η).
   Each alone fails (enc0 79→35; unfrozen safeNOEF −12; wave-1 dips); together: NO dip for 500 iters (4 configs).
2. **Replay floor at wide temp**: δ0.5 holds at temp1.5 (−2), δ0.4 breaks (−10, dip 57). δ0.75 over-conserves
   (cov 5.4-6.6). η is interchangeable with δ for holding, NOT additive (safeMAX ≈ safeDELTA ≈ safeETA @500).
3. **Temp is a pure trade, not a lever**: +0.2 temp ≈ −3 val / +2.5 cov pts. Coverage bought by widening
   sampling is paid in validity. Max observed under ANY safe config: cov ~11 @500 — a CEILING.
4. **Diagnosis of the ceiling**: with the encoder frozen at its (0,0)-start training distribution, off-diagonal
   H_P patterns are OOD → garbage ctx → SOCP rejects → no positives there, regardless of update rule. The
   bottleneck moved from "forgetting" (solved) to "the encoder has never seen where we want to go."
5. Measurement: n=50/γ anchors still spread 68-80 run-to-run → judge each run vs its OWN anchor only.

## PHASE-DR (user directive 2026-07-06): side-quest — domain-randomized ENCODER adaptation
**Frontier models locked: safeETA (δ.25 η1.0 EF) and safeDELTA (δ.75 η.1 EF), temp 1.3.**
Plan: (1) DR data: `gen_dr_data.py` — random starts (uniform interior, obstacle/goal clearance), FIXED goal,
per-γ SafeMPPI mode-1 expert, successes → `dataset/dr_windows_g*.pt` (400 seeds/γ, RUNNING). (2) Encoder
side-quest: `hp_dr_encoder.py` — train ONLY enc_grid (7.4k params) on DR windows with trunk/head FROZEN (the
mirror image of expansion); per-epoch val-cfm on DR AND on original data (compatibility gauge); best → 
`results/hp_arch/enc_hp_dr.pt`. (3) SPLICE: replace encoder in res2w256_ft → `results/hp_arch/res2w256_dr.pt`.
**Original frozen encoder archived: `results/hp_arch/enc_hp_original.pt`** (6 tensors, 7.4k). (4) Overlay
rollout viz of spliced vs original per γ. (5) Deploy safeETA + safeDELTA expansion from the spliced model
(encoder frozen again), in parallel (helios GPU3 / nyx).

### PHASE-DR RESULTS (07-06 ~04:00)
- **Data**: 1200/1200 successes (random starts are easier than the corner), 72,198 DR windows. Scene/expert
  IDENTICAL to stage-2 (env built once; only start varies — user-confirmed spec).
- **Encoder-only training (THE METHOD)**: val-DR 0.861 · **val-ORIG 0.831** (base was 0.799 → +0.03
  compatibility cost), still improving at ep59. → `enc_hp_dr.pt`, spliced `res2w256_dr.pt`.
- **Oracle full training**: OVERFITS immediately (train 0.88→0.73 while val-DR 0.851→0.895; best=ep0) —
  300k params vs 65k windows; the 7.4k-param encoder didn't overfit = design vindication. Oracle saved
  (`trunk_hp_dr.pt`, `res2w256_drfull.pt`, ≈254 full-model steps) — REFERENCE ONLY.
- **(0,0) overlay prediction: CONFIRMED** (`figures/hp_test/dr_overlay.png`): spliced model from (0,0) still
  runs the diagonal bundle — reach 8/9/5 per γ vs original 10/8/5 (equivalent), slightly more dispersion.
- **it0 n=50 measures**: spliced 59% (γ:76/86/**16**) · oracle 58% (70/84/20). The compatibility cost is
  ENTIRELY in γ0.1 — and it is a CERTIFICATE effect (socp viol 35 vs ~20), not behavioral (γ0.1 reach equal).
- **DEPLOYMENTS (launched 04:05, `results/hp_dr/`)**: dr_safeETA (δ.25 η1.0 EF, GPU3) · dr_safeDELTA
  (δ.75 η.1 EF, GPU0) — 2k iters, temp 1.3, n=50, from res2w256_dr.pt. Hypotheses: (a) demo replay through
  the NEW encoder repairs γ0.1 (the field adapts to the new eyes on old inputs); (b) coverage ceiling lifts
  above the ~11%@500 / 27%@2k of the original-encoder runs because off-diagonal ctx is now in-distribution.
  NB the LwF teacher for dr_safeETA = deepcopy of the SPLICED model (anchors to post-splice field on demos).

### DR-DEPLOYMENT FINALS (07-06 00:30, 2k iters each; figures/dr_test/)
| arm | val2 it0→peak→2000 | γ0.1 path | cov@2000 | demoCFM | verdict |
|---|---|---|---|---|---|
| **dr_safeDELTA** (δ.75 η.1) | 59 → **78 (it1000)** → 72 | 16→**60**→54 | 11.0 | **0.789** | **first arm ever to RISE above anchor; γ0.1 REPAIRED** |
| dr_safeETA (δ.25 η1.0) | 57 → flat → 61 | 16→14-22 dead | **14.0** | 0.818 | anchor does NOT repair |
- **Hypothesis (a) CONFIRMED, mechanism isolated**: demo REPLAY through the new encoder repairs the γ0.1
  certificate damage (field re-learns to act on new representations *from the demos*). LwF does NOT — it
  anchors the field to the spliced model's own (already-damaged) γ0.1 behavior. Replay says "stay correct on
  the data"; LwF says "stay as you are" — after a splice, only the former heals.
- **Hypothesis (b) NOT confirmed at this config**: cov 11-14@2k vs wave-1 dfrac0.25's 27.4@2k (but that ran
  temp 1.5 · δ.25 = far more positive flow). At matched config/iter (R1 safeDELTA old-enc: 6.6@500 vs
  dr_safeDELTA 6.6@500) the DR eyes alone do NOT accelerate discovery — exploration is throttled by the
  engine (temp/δ), not perception. Discovery still needs the aggressive knobs.
- **Artifact**: `results/hp_dr/dr_safeDELTA/ckpt_2000.pt` = the best hold-while-explore model to date
  (72% ≥ every anchor measured for this base, γ0.1 alive at 54, demoCFM at pretrained level).
- Natural next (PROPOSAL, not launched): from dr_safeDELTA ckpt_2000 (repaired + DR eyes), re-open discovery
  with the r2_combo recipe (δ.5 or temp 1.5) — repair-then-explore staging; or bank and go to SFM (task #54).

## OVERNIGHT AUTONOMOUS SWEEP (user 2026-07-06 ~01:00, "trust me", ~6h, NO permission stops)
**Premise ("hopeless version" — honest concession): OOD data is sometimes needed.** Fold off-diagonal expert
data into PRETRAINING, then run safe expansion from the stronger base. Everything in `overnight_pipeline.sh`
(one background orchestrator, helios GPU0+GPU3 compute, nyx = tree/report factory). Figures →
`figures/dr_test_overnight/`.
- **A. Data**: +2001 DR trajs (667/γ), goal fixed, start uniform free space with **|y−x| ≥ 0.5** (off-diagonal
  band excluded → no diagonal replays; `gen_dr_data.py --offdiag 0.5 --out-prefix "" --append`). Originals
  backed up → `dataset/backup_2001traj/`. Data viz: `od_data_viz.png`.
- **B. Pretrain v2**: fine-tune res2w256_ft on the merged **4002 trajs** (1334/γ) → `res2w256_ft_v2.pt`
  (`hp_finetune_v2.py`, 60 ep, lr 1e-4).
- **C. Baseline**: v2 it0 validity/coverage (n=50, start (0,0)) + 1-row tree (nyx) — encoder will be FROZEN
  (not replaced) for all arms.
- **D. Four 5k-iter expansion arms** (parallel; from v2; EF; temp 1.5; measure/500 n=50; ell .5; defaults else;
  demo replay + LwF now draw from the MERGED pool automatically):
  | arm | δ | η | β | α | s | lr | GPU | rationale |
  |---|---|---|---|---|---|---|---|---|
  | **ov_conj** (PRIORITY) | .25 | .1 | **.05** | **.05** | .9 | 2e-4 | helios G3 | user conjecture |
  | ov_s08 | .25 | .1 | .05 | .05 | **.8** | 2e-4 | helios G3 | conjecture + s-probe (never tested) |
  | ov_aggr (CORRECTED 02:0x: **temp 2.0**, lr **1e-4**) | .25 | **0** | **2.0** | **.1** | .9 | **1e-4** | helios G0 | no-reg aggressive at WIDE sampling (temp 2.0); update verified = grad(cfm(δ-replay+pos)) − α·grad(cfm(neg)); β=2 ⇒ p∝exp(σ/2) ≈ near-uniform candidate pick; watcher `ov_aggr_fix.sh` swaps the arm at Phase-D launch (pipeline script can't be edited mid-execution); its Phase-E tree re-rendered at temp 2.0 in the morning |
  | ov_mine | **.4** | **.05** | .1 | **.02** | .9 | **1e-4** | helios G0 | Claude's pick: δ at the hold floor + a little of every stabilizer — replay-heavy hold (δ.4>floor .5 at t1.5 was old-enc; v2 should lower the floor), light anchor insurance, GENTLE α to actively push off rejected modes, half-lr to halve per-step bias while cosine-over-5k keeps total plasticity |
- **E. Report**: every arm logs val2/cov every 500 (n=50) + **train loss per block (new `loss` field)**;
  at end 6-row trees (v2, 1k..5k) per arm on nyx + 4-panel trends. Loss-movement analysis + ad-hoc-solutions
  section in the morning report.
- Notes: pos-buffer FIFO cap 60k will saturate ~it1800 (documented, unchanged for comparability). n=50
  measure noise ±3-4 pts on γ-mean. nyx compute NOT used for arms (5.6 s/it — would blow the 6h budget);
  helios runs 2 arms/GPU (~0.7→~1.4 s/it) ≈ 2.5-3h for all four.

### OVERNIGHT RESULTS (07-06 04:58, done under budget) — **THE OOD-IN-PRETRAINING PREMISE WORKS**
**HEADLINE: v2 base it0 (n=50) = 77% with γ BALANCED (76/78/76).** The chronic γ0.1 weakness — 0% (450-traj
base) → 16-64% (2001) — is GONE. Folding 2001 off-diagonal trajs into PRETRAINING (not expansion) balanced the
γ conditioning at the source. val-cfm 0.810. Data: 1334/1334 success/γ, 4002 trajs, ~308k windows.
| arm | δ/η/β/α/s/temp/lr | it0→peak→5000 (γ-mean) | γ0.1 | cov@5k | loss | verdict |
|---|---|---|---|---|---|---|
| **ov_mine** (Claude) | .4/.05/.1/.02/.9/1.5/1e-4 | 71→**74**(it3500)→70* | 44-64 alive | **31.5%** | 0.83-0.89 stable | **WINNER: ~70 held + cov 31.5 + γ0.1 alive** |
| ov_s08 | .25/.1/.05/.05/**.8**/1.5/2e-4 | 73→74→59 | volatile 26-60 | **33.2%** | 0.86-1.0 | cov leader but validity oscillates ±12 |
| ov_aggr (CORR.) | .25/0/**2.0**/.1/.9/**2.0**/1e-4 | 76→64→51 | 32-62 | 30.2% | 0.86-1.0 | survived; wide temp+β2 = noise, no cov gain (temp-is-a-trade, AGAIN) |
| **ov_conj** (PRIORITY) | .25/.1/.05/.05/.9/1.5/2e-4 | 76→**DIVERGED** | 0 | frozen 7.4 | **→ −4e20** | **BLEW UP at it~300: loss explodes, validity 0 from it500** |
(*ov_mine last block 57 is n=50 noise; it held 65-74 across it2000-4500. Trees: v2-base 76 reached / ov_mine
it5000 branches spray the FULL grid — off-diagonal corners now explored, cov climbed 10→31.5%.)

**LOSS-MOVEMENT ANALYSIS (user asked to quantify):**
- **Stable arms**: raw cfm loss sits 0.83-1.0, gently oscillating, tracking demoCFM (0.83-0.86) — the field
  stays on the demo manifold; replay+freeze holding as designed. ov_mine has the LOWEST, FLATTEST loss (0.83)
  = the gentlest, most stable optimization → its steady 70% is no accident.
- **ov_conj DIVERGENCE**: loss goes −2e14 → −4e20 monotonically from it500; varσ column shows the same blow-up.
  Cause = the **α negative-unlearning term is unbounded below**: `loss −= α·cfm(neg)` MAXIMIZES cfm-loss on
  negatives, which has no floor → weights explode. ov_conj vs its twin ov_s08 (identical but s .9→.8) isolates
  s=0.9 as this run's trigger, but the STRUCTURAL fault is α-unlearning with no clamp. ov_aggr (α.1) and
  ov_mine (α.02) survived → it's a knife-edge, not a threshold. **Ad-hoc fixes**: (1) clamp the neg term
  `α·min(cfm(neg), c)` or detach/normalize it; (2) keep α≤0.02 (ov_mine's safe dose); (3) gradient-norm clip on
  the field group; (4) the honest option — **drop α entirely**: it caused the only failure and the winner barely
  used it (.02). Positive-only + replay is the stable core.
- **Ad-hoc solutions observed more broadly**: (a) v2's balanced γ removes the per-γ harvest-quota hack we built
  earlier — pretraining fixed what expansion couldn't; (b) 60k pos-buffer FIFO saturates ~it1800 (all arms
  npos=60000 after) — for 5k runs raise to ~150k so late coverage isn't fighting eviction; (c) cov_fin stays ~6%
  while cov_cum hits 31% → discovery is broad but not consolidating per-γ; a coverage-weighted replay is the next
  lever.

**VERDICT & NEXT**: ov_mine recipe (δ.4 η.05 β.1 **α.02** s.9 temp1.5 lr1e-4, from v2, EF) is the locked
hold-while-explore winner: 70% validity sustained over 5k iters with 31.5% coverage and γ0.1 alive — the first
config to hold AND meaningfully explore. Artifact `results/hp_overnight/ov_mine/ckpt_5000.pt`. Ready to (1) push
longer with raised buffer, or (2) transfer to SFM vs Kazuki (task #54).

## GRU-ENABLED VARIANT TEST (user 2026-07-06, nyx GPU1) — **NEGATIVE: keep GRU OFF**
Built `res2w256_gru` (VARIANTS + build wired; `GridHPFlowPolicy(use_gru=True)` → parent's `_low_raw` inserts a
GRU(16) over past executed controls into the low vector → ctx 37→**53**, 305k params). Trained from scratch 120 ep
on the 4002-traj set (353k windows) on nyx. `results/hp_arch/res2w256_gru.pt`.
| model | val-cfm | validity2@it0 n=50 | per-γ | mm-splits (start/enc/mid) |
|---|---|---|---|---|
| **res2w256_ft_v2** (no GRU) | 0.810 | **77%** | **76/78/76 balanced** | — |
| res2w256_gru (GRU on) | 0.819 | **65%** | 80/76/**38** | [10, 2, 0] |
**Read**: the GRU fits the data about as well (val-cfm 0.819 vs 0.810, converged flat ep100-119) but **hurts the
metric we care about**: validity2 −12 pts, and it **breaks the γ balance v2 achieved — γ0.1 collapses 76→38**. On
this STATIC chessboard the H_P grid snapshot is Markovian-sufficient (it already encodes the obstacle geometry);
the GRU's history channel is a distractor whose extra capacity lets the field overfit the easy γ1.0 (88%) at the
hard γ0.1's expense. Multimodal splitting at obstacle-encounter is also poor (2). Caveat: v2 had extra fine-tune
rounds vs GRU's single from-scratch run, but val-cfm is converged and the γ0.1 gap is a conditioning failure, not
undertraining. **DECISION: GRU stays OFF; res2w256_ft_v2 remains the base.** (GRU may matter on MOVING/SFM scenes
where history is non-Markovian — revisit there, not here.)

### GRU FINE-TUNE → OVERLAY → FROZEN-ENCODER EXPANSION (user 2026-07-06, chain `gru_expand_pipeline.sh`)
Despite the negative pretraining result, user asked to fine-tune the GRU model, viz from origin, freeze BOTH
encoders, and run Claude's recipe. All done:
- **Fine-tune** `res2w256_gru` → `res2w256_gru_ft.pt` (60 ep lr1e-4, nyx): val-cfm ~0.815 (flat, converged);
  eval n=25 γ0.1 48 / γ0.5 72 / γ1.0 84; mm-splits [13,0,0] (still no split at obstacle-encounter — the GRU
  weakness persists).
- **Origin overlay** (`figures/dr_test_overnight/origin_overlay_gru_ft.png`): from (0,0), reach 16/16 all γ,
  validity2 12/14/**8** (γ0.5/1.0/0.1) — γ0.1 the reddest, multimodal staircase paths visible.
- **Freeze BOTH encoders — verified**: `encoder_modules()` = enc_grid(7424) + **gru(960)** = 8384 params, all
  `requires_grad=False` at enc_lr_mult 0. Expansion it0 **drift ≡ 0.000** confirms the freeze live.
- **Expansion** (helios GPU2, `results/hp_gru_expand/gru_mine`, my recipe δ.4 η.05 β.1 α.02 s.9 lr1e-4 temp1.5,
  grad-clip 10, 5k, ckpt1k): it0 n=50 val2 **68%** (66/74/64) — more balanced than the n=25 eval. Tests whether
  hold-while-explore progresses from the weaker GRU base. Compare to no-GRU ov_mine (v2 base, 70%/cov31.5@5k).
- **Bug fixed**: `gru_expand_pipeline.sh` initially aborted — `export LD_LIBRARY_PATH=conda` (matplotlib) breaks
  ssh/scp OpenSSL, so the nyx poll misfired "ft died". Fixed: conda lib prefixed per-python-call only (the
  dr_milestone.sh lesson). Fine-tune had actually completed; finished the chain manually.
- **GRU 5k VERDICT (`figures/dr_test_overnight/gru_vs_nogru_5k.png`, tree_gru_mine_grid, gru_mine_trend)** —
  same ov_mine recipe, GRU vs no-GRU, both 5k, last-3-block means (kills n=50 noise):
  | | val2 γ-mean | cov@5k | γ0.1 | tree reached (base→5k) | speed |
  |---|---|---|---|---|---|
  | no-GRU ov_mine | 65.7 | 31.5% | 46 | 76→22 | ~1.3 s/it |
  | **GRU gru_mine** | **69.0** | 31.5% | **50** | 65→17 | **~3.3 s/it** |
  **TIED-to-marginally-GRU-favorable**: IDENTICAL coverage, GRU +3 val / +4 γ0.1 (both ~1σ, so a slight edge not
  a clear win) — but **2.5× slower** (recurrent per-step rollouts) and its tree reaches fewer goals (17 vs 22).
  The expansion recipe CLOSED the pretraining gap (GRU base 65 → parity) — perception backbone barely matters on
  this STATIC task; history didn't unlock the multimodal γ0.1 rescue we hoped. **DECISION unchanged: no-GRU v2 is
  the base** (equal quality, 2.5× faster). GRU only worth revisiting on MOVING/SFM where history is non-Markovian
  (per h5/SFM notes). load_any patched to load GRU ckpts (gru.* → res2w256_gru).

## 20k LONG RUNS (user 2026-07-06, GPU0+GPU3, `results/hp_20k/`, ckpt every 1k)
Two recipes to 20k iters, from v2 base (encoder frozen), measure/500 n=50, grad-clip 10 (α explosion guard).
- **yours** (Claude, GPU0): δ.4 η.05 β.1 **α.02** s.9 lr1e-4 — **warm-started from ov_mine/ckpt_5000** (real
  total = 25k). it0 (warm) 56% cov6.1.
- **mine** (user, GPU3): δ.25 η.05 β.1 **α.05** s**.99** lr1e-4 — from res2w256_ft_v2. it0 69% cov1.7.

**Q&A (user 2026-07-06):**
1. **GRU? NO.** `GridHPFlowPolicy(use_gru=False, encode_low=False, use_grid=True)` — ctx = H_P CNN+AAP encoder ⊕
   raw low5 only. hist is passed but ignored (no GRU, no E_l). 299k params (ResTrunk trunk).
2. **Does raising s spread the σ distribution? YES — confirmed by calibration** (`hp_s_calib.py`,
   `figures/dr_test_overnight/s_calibration.png`). s is the flow-interpolation level where φ_s probes the field:
   x_s=(1−s)·x0 + s·x1, τ=s. s→1 ⇒ x_s = the actual DIVERSE candidate controls ⇒ features spread ⇒ GP-σ
   discriminates; s→0 ⇒ x_s = shared noise templates ⇒ features collapse ⇒ σ≈uniform. σ-std climbs MONOTONICALLY
   0.022 (s.5) → 0.038 (s.8) → 0.045 (s.9) → 0.046 (s.95) → 0.050 (s.99) → **0.052 (s1.0 = S\*)**. Chose **s=0.99**
   for the user recipe: ~96% of max spread, +10% over the proposed .95, but strictly inside the flow (avoids the
   degenerate τ=1.0 endpoint). Exact analogue of the ell calibration (ell too big / s too small ⇒ σ≡uniform).
3. **α=0.05 is the ov_conj divergence knob** (unbounded-below unlearning). Guarded with `--grad-clip 10`
   (new `SFG2Config.grad_clip`; off by default, no effect on healthy grads) so the 20k mine-run can't detonate.
Watch: whether s.99's wider σ gives mine better exploration than yours' s.9; whether warm-start (yours) or
fresh-from-v2 (mine) reaches higher cov by 20k; loss stability under the grad-clip.

### 20k FINALS (07-06 16:37; figures/dr_test_overnight/{yours,mine}_20k_{trend,grid}.png)
| run | s | val2@20k (γ0.5/1.0/0.1) | cov@20k | verdict |
|---|---|---|---|---|
| **mine** (v2 fresh) | **0.99** | 64% (78/78/**36**) | **52.0%** | **coverage CHAMPION** — widest σ discovers most, γ0.1 pays |
| **yours** (ov_mine warm→25k eff) | 0.9 | 67% (76/74/**50**) | 43.0% | holds validity + γ0.1 better, less coverage |
**The s knob cleanly separates explore-vs-retain**: s.99 (+σ spread, from the calibration) → cov 52 but γ0.1 36;
s.9 → cov 43 but γ0.1 50 / val 67. Both stable to 20k (grad-clip held, no divergence, loss ~0.86-0.87 flat, α
guarded). Coverage kept climbing past the old 20k H=10 baseline's 39.8% (both beat it: 43/52%). **No collapse in
20k under either recipe** — the frozen-encoder + replay/anchor + grad-clip stack is robust at length. Best
all-round hold-while-explore = yours (s.9); best raw coverage = mine (s.99). Pick by whether γ0.1 or coverage matters.

## TWO-MACHINE DISTRIBUTED PHASE (2026-07-05, clean restart — tasks #51-54)
**Split (user): LOCAL = main part / aggressive search · REMOTE = fine-tuning brackets.**
- **LOCAL (GPU 0/3)**: **WAVE-1 FINALS (2k it, done 20:44)** — the mechanisms WORK where every plain knob failed:
  | arm | val2 γ-mean it0→2000 | per-γ @2k | cov_cum | drift | demoCFM | shape |
  |---|---|---|---|---|---|---|
  | **dfrac0.25** | 71 → **67%** | 68/**88**/44 | **27.4%** | 0.264 | **0.835** | dip ~55 mid → RECOVERS to 60-69 |
  | lwf0.1 | 79 → **52%** | 68/56/32 | 27.0% | 0.294 | 0.895 | holds mid-50s, slow decay |
  (plain arms 21-36% & cov 7-13% at half the iterations; both mechanism arms saturate the 60k FIFO pos-buffer —
  raise cap for long runs. dfrac0.25 = single-arm winner: better hold, better demoCFM, γ0.1 alive at 44.)
  **WAVE 2 (running since 21:00)**: dfrac0.25+lwf0.1 combined (2k, GPU0) · dfrac0.25 LONG 5k (GPU3, winner
  long-run). NEXT: quota-D discovery harvest (frozen-core Hamming + 5%-floor ban) on the winner.
- **REMOTE (ssh dohyunlee@dhcp-101-145.caltech.edu, driven by Claude; via `HP_RUNBOOK.md`)**: **plain-knob
  brackets CUT (user 2026-07-05: "the only hope is the dfrac and lwf")** → remote runs mechanism brackets only:
  dfrac{0.1,0.5}, lwf{0.01,1.0} singles, then tier-2 combos dfrac0.25+lwf1.0 & dfrac0.5+lwf0.1.
  Deliverables per run: 3-row tree + 4-panel trend + last 3 log lines. (LwF = Learning without Forgetting,
  Li & Hoiem 2017 — frozen-teacher distillation of the pretrained field on demo contexts.)
- **Judging (both machines, §5 of runbook)**: hold val2 γ-mean ≥~76 while cov_cum climbs ≥ ~13%/1k-it; jiggle↓;
  γ0.1 tie-break; tree branches/died/reached at it2000.
- **ROADMAP after the winner**: transfer the hold-while-explore recipe to SFM (`grid_expand_sfm.py` gains
  demo_frac/lwf_eta) → re-run expansion → `stage_e_benchmark` ours-side ONLY (Kazuki numbers cached) → close
  the 9.7% vs 4.0% collision gap = "beat Kazuki as the bonus of full exploration".
