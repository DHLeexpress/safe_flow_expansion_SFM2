# Coverage / Validity / Diversity for trajectories

> Spec: "원문처럼 coverage-validity 를 비교할 수 있는 걸 그리고 싶은데 우리는 체스판 예시처럼 한 번에
> generate 할 수 있는게 아니니깐, 너가 잘 생각해봐야겠어."

ACTFLOW's chessboard measures coverage on a 2-D histogram of the *design itself*. Our design `U∈R^{T×2}` is too
high-dim to histogram directly, and a trajectory's "mode" is what we actually care about. So we measure coverage on
a **low-dimensional trajectory descriptor** that encodes the homotopy class + clearance — the thing that makes
left/right/middle distinct. This recovers a chessboard-style coverage curve in trajectory space.

---

## 1. Trajectory descriptor `d(U)`  (code: `src/descriptors.py`)

For env `c`, roll out and summarize the trajectory by **how it passes each obstacle**:

- **Passing side** `s_j ∈ {-1,+1}` for obstacle `j`: sign of the cross product of (goal-direction) × (robot−obstacle
  vector) at the obstacle's closest-approach step. `-1`=pass on one side, `+1`=the other. The vector
  `(s_1,...,s_{N_obs})` is the **homotopy class** (discrete mode).
- **Lateral offset** `ell_j ∈ R`: signed perpendicular distance from the start→goal chord at obstacle `j`'s
  longitudinal position. Continuous coordinate within a mode.

```
ENV-A single  : d(U) = ell_1 ∈ R            (sign of ell_1 = left vs right)
ENV-B gap     : d(U) = (ell_1, ell_2) ∈ R^2 (macro-mode ∈ {left-of-both, through-gap, right-of-both})
```

**Macro-modes** (the human-meaningful "left/right", "left/mid/right"):
- ENV-A: `LEFT = ell_1<0`, `RIGHT = ell_1>0`.
- ENV-B: from the pair of passing sides → `{LEFT, GAP, RIGHT}` (GAP = between the two obstacles).

---

## 2. Validity (verifier pass-rate)
```
Validity(theta) = (1/K) Σ_{U~q_theta} v_cert(U, c)      # fraction of policy samples that are certifiably safe
```
ACTFLOW toy went 76.00% → 95.89%. We report this per round.

## 3. Coverage (descriptor-bin occupancy of the SAFE set) — the chessboard analog
1. Build `Omega*` once per env: densely sample a **broad** proposal (wide-σ MPPI / box-uniform `U`), keep
   `v_cert=1`, histogram their descriptors → the set of **reachable-safe bins** `B*` (paper: 100×100 hist). Also
   record which macro-modes are populated (ground-truth: ENV-A has 2, ENV-B has 3).
2. For the policy: sample `K`, keep `v_cert=1`, histogram descriptors, threshold each bin at `tau` (paper `tau=0.01`
   of the K samples) → populated bins `B_theta`.
```
Coverage(theta) = | B_theta ∩ B* | / | B* |          # in [0,1], chessboard-style
ModeCoverage(theta) = (# macro-modes with policy prob ≥ p_min) / (# macro-modes in Omega*)
```
`ModeCoverage` is the headline number ("did it find left AND right AND middle"); `Coverage` is the fine-grained
continuous version that yields a smooth curve like the paper's 1.16%→94.27%.

## 4. Diversity (Vendi score)
```
Vendi(S) = exp( - Σ_i λ_i log λ_i )      # λ_i = normalized eigenvalues of the sample kernel matrix
```
Kernel = RBF on descriptors `d(U)` (paper used RBF on embeddings). Reported on the SAFE samples per round; rises as
the policy spreads across modes. A unimodal policy → Vendi≈1; bimodal balanced → ≈2; trimodal balanced → ≈3.

## 5. The plots (code: `src/plots.py`)
1. **Coverage–Validity curve** vs round: twin-axis line plot (coverage% left, validity% right), one figure per env.
   Overlay baselines `REC-F` (resample-from-policy, no tilt: `beta→∞`) and `REC-NF` (no fine-tune) to reproduce the
   paper's "naive regeneration can't cross the boundary" story.
2. **Mode-coverage / Vendi** vs round (companion curve).
3. **Multi-modal trajectory overlays** at rounds {0, mid, final}: many sampled trajectories on the obstacle map,
   colored by macro-mode (left/right/[mid]), obstacles as circles+margin, start/goal markers — reusing the repo's
   matplotlib style (green/red/dark-blue palette, `alpha≈0.15` clouds). Shows the seed's single leaf opening into
   all leaves.
4. (optional gif) the overlay animated across rounds — the "virus spreading to all cells" analog.

## 6. What "success today" looks like
- ENV-A: seed = right-only (ModeCoverage 1/2, Vendi≈1); final = both sides (ModeCoverage 2/2, Vendi≈2), Coverage and
  Validity curves rising and saturating.
- ENV-B: seed = one side; final = left/gap/right all generated (ModeCoverage 3/3, Vendi≈3), the gap mode threading
  the narrow passage while remaining `v_cert`-safe at the non-conservative `gamma` — the concrete demonstration that
  Safe Flow Expansion learns *less conservative* multi-modal behavior.
