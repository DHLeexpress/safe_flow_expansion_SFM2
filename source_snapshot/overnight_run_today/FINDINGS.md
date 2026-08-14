# FINDINGS — what actually happened building SafeFlow Exploration

A faithful, warts-and-all account of developing the paradigm and especially the **Active Flow Expansion**
(ACTFLOW Eq.9/Eq.10) part. Numbers below are from the real runs in `results/`. Read alongside
`design/SAFEFLOW_GLOSSARY.md` (definitions) and `design/VERIFIER.md`.

TL;DR — The headline demo is **real**: from a one-leaf conservative seed the FM policy learns *all* homotopy
modes (left/right; left/gap/right) and coverage/validity/diversity move the ACTFLOW way. But **mode discovery is
carried by a broad "surrounding" proposal, not by the σ-acquisition**, and the specific value of Eq.9/Eq.10 is
*not yet proven* (the σ-tilt currently behaves near-uniformly). Details and honest caveats follow.

---

## Paradigm as committed
design = whole control sequence `U∈R^{40×2}`; valid = DTCBF-certifiable-safe **and** goal-reaching;
verifier = certificate optimization; Eq.10 σ = GP posterior variance over the FM noised-flow feature `φ_s`;
Eq.9 = uncertainty-tilted exploration; UpdateFlow = signed CFM grad (learn safe, unlearn unsafe).

---

## Finding 1 — My first verifier was geometrically wrong; the fix is what makes 3-D trivial
- **What I tried first:** per obstacle, one **time-invariant** separating hyperplane over the whole horizon
  (angle-swept LP). Result: **`0/4000` valid** — it rejected everything.
- **Why:** a path that goes *around* an obstacle has its obstacle-relative vector rotate ~180° (robot in front →
  beside → behind). **No single static half-space separates a wrap-around trajectory.** This is exactly why the
  repo's `barrier.py` recomputes the normal each step.
- **Fix:** the **distance DTCBF** `h_{j,t}=‖p_t−c_{j,t}‖−(r_j+r_robot)` with per-step normal `∇h`. Certifiable iff
  `h≥0` ∀t and `∃γ≤γ_max: h_{t+1}≥(1−γ)h_t`. Sound, closed-form, vectorized.
- **3-D answer (your question "무리일지"):** the distance check is **O(dim)** with *no* SOCP and *no* normal
  search — dimension-agnostic. The verifier is **never** the bottleneck; FM sampling/rollouts dominate. So 3-D is
  fine. After the fix: seed `2000/2000` valid; broad search recovers all modes.

## Finding 2 — THE central result: pure Eq.9 tilting cannot discover a disconnected mode
- Implemented Eq.9 faithfully (sample `N` **from the FM policy**, tilt by `exp(σ/β)`, resample, verify), 60 rounds,
  single obstacle. Result: **modecov stuck at 0.50 forever**; LEFT never appeared (`probs=['0.00', …]`).
- **Why (math, then measured):** the exact maximizer of `E_q[σ]−β·KL(q‖p_θ)` is `q* ∝ p_θ·exp(σ/β)`. If
  `p_θ(LEFT)≈0` then `q*(LEFT)≈0` regardless of σ — reweighting can't create mass the policy never proposes. I
  probed how fat the tails must get:

  | proposal temp / churn | LEFT-valid / 8000 | overall valid |
  |---|---|---|
  | 1.0 / 0.0 | **0** | 4051 |
  | 2.5 / 0.3 | **0** | 845 |
  | 4.0 / 0.5 | **4** | 332 |

  In an 80-dim control space the RIGHT-pretrained flow essentially **never** emits a coherent valid LEFT maneuver.
- **Why ACTFLOW's chessboard escapes this and we don't:** their pretrained model is a Gaussian (everywhere-positive
  tails) in a **2-D** design space, so `exp(σ/β)` can amplify a tiny tail into a far cell. Our valid set is
  genuinely **disconnected** (can't deform left-pass → right-pass without hitting the obstacle), and the practical
  flow sampler concentrates too hard for the tail to bridge it.

## Finding 3 — Resolution (and an honest deviation from ACTFLOW)
- Eq.9's argmax is over **all** `q`; finite `β` *permits* deviation from the prior. So the candidate pool now
  includes a **broad "surrounding" proposal** (SafeMPPI-style wide lateral sampler) alongside FM samples; the σ-tilt
  **selects** among them and the verifier filters; only verified-valid samples train the FM.
- With this: single-obstacle **modecov 0.50→1.00, cov 0.29→0.79**; gap → **all three modes**.
- **Honest framing:** this is a **deviation from pure ACTFLOW** (which samples only from the flow), and *the broad
  proposal does most of the mode-discovery work*. Justification: (a) legitimate importance support for `q*`; (b) the
  FM is the only thing learned and measured — coverage is computed on the FM's own `temp=1` samples, so it rises
  only if the FM genuinely *learns* the modes from the distilled buffer; (c) matches your spec's "surrounding
  으로 constrained 된 상황." Defensible, but a design choice — not a faithful reproduction.

## Finding 4 — The uncomfortable caveat about Eq.9/Eq.10's *actual* contribution
The Eq.10 machinery is **mechanically correct** (diagnostics prove it) but its **practical influence on selection
is weak** in this setup:
- **D1 cold-start:** `σ≡1.000` at round 0 (empty buffer, normalized kernel) — the chessboard iteration-0 behavior. ✓
- **D2 shrinkage:** buffer σ (≈0.086) < fresh σ (≈0.171) early. ✓
- **D7 ESS:** ≈N at cold start. ✓
- **But:** σ **collapses to ≈0.02 by round 10** and stays there (once the buffer covers the feature space, every
  candidate looks "known"), and **ESS stays ≈1.0 all run** — with `β=1/13` and σ-spreads ~0.01, `exp(σ/β)` is nearly
  flat, so selection is ≈ **uniform** over (FM ∪ broad). The real drivers are the broad proposal + verifier +
  balanced replay.
- **Implication (honest, open):** in the current setup the σ-acquisition's marginal value over uniform selection is
  **unproven** — but per the decision to drop ablations, we are *not* chasing a REC-F/REC-NF comparison. Instead, if
  we want σ to actually drive exploration, the levers are: smaller `β`, a lengthscale-tuned/longer-lived σ, or an
  **FM-only candidate pool** (so σ over an evolving narrow set discriminates). Noted as a known limitation, not a
  blocker — the multimodality + coverage results stand on their own.

## Finding 5 — Expansion dynamics: two real tradeoffs and their fixes
- **Coverage–validity tradeoff.** When the FM first learns the 2nd mode, the flow *bridges* modes and leaks mass
  into the collision gap → **validity dropped 0.51→0.28** (α=0.005). Raising the negative-unlearning weight
  **α→0.12–0.15** pushes mass out of the invalid gap and restores the ACTFLOW shape (**validity recovered to
  0.58–0.64**, above the seed, while coverage stayed high). α is the knob that buys "validity rises too."
- **Narrow modes get out-competed.** The GAP (middle) mode was *discovered then collapsed* (`G: 0.029→0.003`) — it's
  a tiny target, under-represented in the buffer, so mass migrated to easy L/R. Restored to a stable ~2–3% by
  (a) oversampling the central band in the surrounding proposal, (b) **mode-balanced UpdateFlow** (inverse-frequency
  per-sample weights). Another deliberate deviation from ACTFLOW's uniform replay, justified by unequal mode sizes.

---

## Final results (production runs)

| env | coverage | validity | mode-coverage | Vendi | notes |
|---|---|---|---|---|---|
| **single** seed → final | 0.29 → **0.79** | 0.51 → **0.64** | 0.50 → **1.00** | 1.7 → 4.4 | LEFT+RIGHT learned |
| **gap** seed → final | 0.23 → **0.82** | 0.35 → **0.39** | 0.33 → **1.00** | 1.7 → 5.7 | LEFT+GAP+RIGHT; GAP ~2–3% |

Both: coverage and mode-coverage rise to full; validity stable/up (single) or modestly up (gap); the multi-modal
overlays show the seed's single leaf opening into all leaves (`figures/*_overlay_*.png`).

## What is solid vs. what is unproven
- **Solid:** verifier (+ clean 3-D story), end-to-end loop, Eq.10 diagnostics, the multimodality demonstration,
  the α-controlled coverage/validity behavior.
- **Unproven / honest gaps:** (1) σ-acquisition's value over no-tilt — needs REC-F/REC-NF ablations (σ collapses,
  ESS≈1 today); (2) mode discovery currently leans on the broad proposal, not Eq.9; (3) GAP retention and validity
  recovery rely on design choices (α, mode-balancing, gap-targeted proposal) that depart from vanilla ACTFLOW.

## Recommended next steps
1. **Next phase:** swap the toy scene for pedestrian data and compare against Mizuta — same four stages, same
   metrics, same GIFs (SafeMPPI ruler → FM field → expansion).
2. Make σ informative (optional, no ablation): smaller `β`, tune/anneal lengthscale, or restrict candidates to FM
   samples so the uncertainty actually steers selection.
3. Optionally raise FM capacity for the gap env to lift validity and thicken the narrow GAP mode.

## Stage visualizations (the Figma loop, separated like the old gifs)
- `figures/<env>_stage0_seed_vs_expanded.png` — conservative seed → less-conservative multimodal (vs candidate polytope)
- `figures/<env>_stage1_safemppi_ruler.gif` — SafeMPPI sample-then-reject under the (1-γ)^i ruler (data engine)
- `figures/<env>_stage2_fm_field_certified.gif` — the FM generative field + ruler + verified-polytope faces
- `figures/<env>_stage3_safeflow_expansion.gif` — the loop: single leaf → all modes; coverage/validity/mode-cov rising
See `STAGES.md` for a plain-language, fully-defined walkthrough.
