# overnight_run_2026-07-01 — Windowed FM policy + Safe Expansion (clean restart)

A SOTA-diffusion/flow-matching-policy-style rebuild of Pillars 4–5, run **end-to-end for two obstacle
configurations**:

| scene | layout | SafeMPPI expert behavior | what expansion should add |
|---|---|---|---|
| **gap** (上下) | two obstacles stacked vertically at (3, ±0.55) r=0.35 | **threads** the gap; γ grades it (γ=0.1 → tight **center**, γ=1.0 → **upper/lower**) | cover the whole verified corridor at every γ |
| **slalom** (左右) | two obstacles sequential: A(2.4,+0.45) B(3.6,−0.45) r=0.55 | **goes around** (sweeps above/below the pair) | discover the **weave** / tighter certified modes between them |

Both use a **POINT robot** (`r_robot=0`, which matches the `sm=0` best_area_mode4 planner — it plans against
raw disks and never sees a robot radius; NO `safety_margin`), and the **original ieee unbounded max-margin
verifier** (`make_variable_faces`) — the verifier *expands* its fitted polytope to certify what SafeMPPI's fixed
nominal polytope is too conservative to.

## Why windowed (the fix)
The 2026-06-30 build generated a whole ~80×2 trajectory at once (~160-D) → the verifier discarded ~96% of
samples. Here the FM predicts short **control windows** `U_{t:t+H_pred}` (H_pred=10) and is deployed
**receding-horizon** (execute H_exec=1, replan). 100 episodes × 80 steps → ~10k–15k windows; small per-sample
dimension → far higher validity, exactly like diffusion/flow-matching action-chunk policies.

## Observation = what the SafeMPPI expert sees (`polar_grid.py`, `local_frame.py`)
Robot-centered, **goal-aligned** conditioning:
- **polar polytope-occupancy grid** `[3, N_θ=16, N_r=12]` (R=3=barrier_activation_radius): channels =
  occupancy / polytope_mask (`H_P≥0`) / clipped `H_P`, computed from the exact `build_polytope_v2` the planner uses.
- **low-dim state** `[goal_dist, v·e_g, v·e_lat, a_prev·e_g, a_prev·e_lat, γ, prev_action_valid]`.
- **target** = MPPI planned window `U_{t:t+H_pred}` (`adapter._u_prev`, needs `warm_start=True`) rotated into the
  goal-aligned local frame; inference rotates the sampled `U_local` back to world.

## Pipeline (run both scenes)
```
python run_both.py                 # gap + slalom, 100×3 eps, 60-epoch pretrain, 6-round expand, W&B online
python run_both.py --smoke         # fast end-to-end
python run_both.py --scenes gap    # one scene
```
| stage | file | output |
|---|---|---|
| 1 rollout look | `show_rollouts.py` | `figures/<scene>/safemppi_<scene>.{png,mp4,gif}` (di_grid + black-dot trail) |
| 2 dataset | `stage2_build_dataset.py` | `dataset/windowed_<scene>/{train,val,test}.pt` + per-γ rollout plot |
| 3a pretrain | `stage3_pretrain.py` | `results/<scene>/pretrained.pt` + per-γ FM rollouts (W&B live loss) |
| 3b expand | `expansion.py` | `results/<scene>/expanded.pt` + `stage3_before_after.png` + `stage3_comparison.png` (W&B coverage/validity) |

## Modules (swap ideas here — everything is separated)
- `config.py` — scene registry (gap/slalom), horizons (H_pred/H_exec/verifier window), γ grid, verifier knobs, paths.
- `scenes.py` — `make_narrow_gap` / `make_slalom` / `make_single_obstacle` + geometry tests.
- `di_grid_viz.py` — the di_grid renderer (mode-colored accept/reject, executed ✗, **black-dot state trail**,
  blue-nominal / green-verifier polytope toggle) + `mppi_rollout` (records `_u_prev` windows).
- `verifier_polytope.py` — imports the UNMODIFIED ieee `demo_verifier_polytope` (draw_panel/H_grid/check_certificate/
  make_variable_faces); optional replicated `solve_face_bounded_margin` (the m_max tube knob, off by default).
- `local_frame.py`, `polar_grid.py` — the two new featurizers.
- `windowed_policy.py` — `GridLowFlowPolicy(FlowPolicy)` (enc_low + enc_grid → ctx), `fm_rollout` (closed-loop),
  `windows_of`.
- `validity.py` — **swappable**: `collision_free ∧ reaches_goal ∧ verifier_certified` (append performant/comfort later).
- `coverage.py` — **swappable**: spatial-occupancy + behavior-mode + Vendi; Ω* = broad proposal gated by validity.
- `expansion.py` — the verifier-filtered self-training loop (see below).

## Safe expansion loop (`expansion.py`) & the ACTFLOW connection
```
pretrain on MPPI windows
for each round, for each γ:
    closed-loop FM rollouts (exploratory temp) + a broad "surrounding" proposal
    roll out, VERIFY (collision ∧ goal ∧ SOCP-certified)
    keep only certified positives (their windows)
    finetune the FM on  MPPI demos ∪ verified positives
```
**Eq (9)** (ActiveFlowExpansion): `x_{t+1} ~ argmax_q E_q[σ_t(φ_s(x))] − β·KL(q‖p_θ)`. Finite β *permits* the
proposal to deviate from the current policy — we instantiate that deviation with the **broad surrounding
proposal** (the honest finding from `overnight_run_today` is that this, not the σ-tilt, drives mode discovery).
**Eq (10)** GP posterior variance `σ²(x)=k(x,x)−k(x,X)(K+λI)⁻¹k(X,x)` over the noised-flow feature `φ_s` — the
σ-acquisition; in prior runs σ collapsed (≈uniform selection), so this build keeps the **verifier filter + broad
proposal + balanced replay** as the exploration engine and treats the σ-GP as an optional, unproven tilt (a
faithful, documented deviation, not a silent one). The **verifier is the only safety authority** — only certified
windows ever train the policy (`α=0`: rejects are dropped, not unlearned).

## Honest caveats
- **Imitation distribution shift**: a window policy trained on expert/broad windows can drift under its own
  closed-loop rollout; the self-training round (finetune on the FM's OWN certified rollouts) is the DAgger-style
  mitigation. Pretrained validity is low; it rises through expansion (see `stage3_before_after.png`).
- **Point-robot / no safety_margin**: `r_robot=0` matches the sm=0 planner; the demo is a point navigating disk
  C-obstacles. The verifier > nominal gap comes from **fitted vs fixed polytope**, not from inflation.
- **σ-acquisition (Eq 10) is not the driver** here (documented above).
- **DI position-space certificate** for a static scene (never certifies a collision; collision checked separately)
  — the SI-safe framing GOAL.md endorses.
