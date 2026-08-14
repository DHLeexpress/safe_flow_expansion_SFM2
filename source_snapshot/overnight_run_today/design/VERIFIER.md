# The Verifier as an optimization problem (spec 3.2)

> "verifier 은 optimization problem 일 것 같아. 어떠한 한 polytope with corresponding DTCBF 가 FM policy 를
> match 하는지 check 해야 하니까. 이게 computationally overhead 인지 — 3d 로 확장하면 많이 무리일지 생각해봐."

**Answer up front:** the verifier is a *small convex feasibility program per trajectory*, not a hard global
optimization. In 2-D it is a sweep of LPs over the separating-normal angle (≈ms). In 3-D it becomes a small SOCP /
sampled-normal feasibility — still ms-scale per trajectory. **The verifier is NOT the bottleneck**; FM sampling and
rollouts dominate. Details below.

---

## 1. What must be certified

Given a sampled control sequence `U`, roll out `xi(U;x0) = {x_t=(p_t,v_t)}_{t=0..T}` and the constant-velocity
obstacle predictions `c_j(t) = c_j(0) + t·dt·v_{o,j}`, radius `r_j`, robot radius `r_r`, margin `m=r_j+r_r`.
We want a witness `(P*, gamma)` — a polytope `P*` (one separating half-space per obstacle, normal `n_j`, the safe
side is `n_j·p ≤ b_{j,t}`) and a decay `gamma ≤ gamma_max` — such that for all obstacles `j` and steps `t`:

```
(containment)   n_j · p_t        ≤ b_{j,t}                       # robot inside P*
(separation)    n_j · c_j(t)     ≥ b_{j,t} + m_j                 # obstacle outside, with margin
(DTCBF decay)   h_{j,t+1} ≥ (1-gamma) h_{j,t},   h_{j,t} := b_{j,t} - n_j·p_t ≥ 0
(unit normal)   || n_j || = 1
```

`v_cert(U)=1` iff this is feasible. The optional velocity look-ahead `eta` term (`h_ho` in `barrier.py`) adds
`- eta·(v_t - v_{o,j})·n_j` to `h_{j,t}` — still **affine in the unknowns** because `v_t` is fixed by the rollout.

### Decoupling per obstacle
The faces don't couple (each obstacle gets its own `n_j, b_{j,·}`), so `v_cert = AND_j feasible_j`. We certify each
obstacle independently → `N_obs` tiny programs, trivially parallel.

---

## 2. 2-D solution: LP sweep over the normal angle

The only non-convexity is `||n_j||=1`. In 2-D parametrize `n_j(theta)=(cos theta, sin theta)`. **Fix `theta`** ⇒
everything is linear in the remaining unknowns `({b_{j,t}}, gamma)` ⇒ a Linear Program:

- variables: `b_{j,0..T}` (T+1) and `gamma` (1) — i.e. `T+2` vars per obstacle;
- constraints: containment (T+1), separation (T+1), decay (T), `0 ≤ gamma ≤ gamma_max`, `h_{j,t} ≥ 0`.

Feasibility LP via `scipy.optimize.linprog` (HiGHS). Sweep `theta` over a grid `Theta` (e.g. 90–180 angles, or
seed angles from the robot→obstacle directions along the trajectory and refine). `feasible_j = OR_{theta∈Theta} LP`.

**Practical shortcut (what `src/dtcbf.py` actually does first):** the tightest separating offset is
`b_{j,t}* = n_j·c_j(t) - m_j` (separation active). Substitute → containment becomes
`n_j·(c_j(t)-p_t) ≥ m_j` (robot strictly closer to its side than the obstacle by `m`), and decay becomes a linear
inequality in `gamma` only. So for each `theta` we (i) check containment in closed form, (ii) solve a 1-D feasibility
in `gamma`. This reduces most queries to vectorized numpy (no LP call) and only falls back to `linprog` when we let
`b` float (for extra slack). Either way the cost is dominated by the angle sweep, fully vectorizable over the batch.

### Complexity (2-D)
Per trajectory: `O(|Theta| · N_obs · T)` flops, vectorizable to a few matmuls. With `|Theta|=120, T=40, N_obs=2`,
that is ~10^4 flops → **microseconds on GPU, sub-ms on CPU**, batched over thousands of candidate sequences at once.

---

## 3. 3-D scaling — is it "무리"? No.

In 3-D the normal lives on the sphere `S^2` (2 DOF). Two options, both cheap:

1. **SOCP relaxation.** Replace `||n||=1` with `||n||≤1` (convex). Feasibility becomes a small Second-Order Cone
   Program in `(n_j∈R^3, {b_{j,t}}, gamma)`. Solvable by `scipy`/`clarabel`/`ecos` in ~ms. The `≤1` relaxation is
   exact for separation/containment feasibility (scale-invariant up to `h≥0` which we normalize).
2. **Sampled / seeded normals.** Replace the angle grid with a Fibonacci-sphere set of `~200` candidate normals, or
   seed from the rollout's robot→obstacle directions and do one local refine. Then it is again an LP per normal.

Cost scaling vs 2-D: the per-program size grows only by the +1 normal dimension; the normal *search* grows from a
1-D grid to a 2-D set (`|Theta|` → `|S^2 samples|`, ~120 → ~200). So roughly **2–3× the 2-D cost** — still ms-scale,
still dwarfed by FM ODE sampling. The geometry (separating hyperplanes, affine DTCBF) is dimension-agnostic; nothing
combinatorially explodes. **Verdict: 3-D is fine for the verifier.** The real 3-D costs are elsewhere (longer state,
more obstacles, FM net size), not the certificate.

| | per-traj verifier | dominant cost in the loop |
|---|---|---|
| 2-D | LP/closed-form sweep, ~µs–sub-ms (batched) | FM ODE sampling + UpdateFlow grad steps |
| 3-D | SOCP / sampled-normal, ~ms (batched) | same — FM net + rollouts |

---

## 4. Two verifier "levels" and how we use them

- **`v_collision` (ground truth, cheap):** min clearance over the rollout `≥ 0`. Necessary condition; used as a fast
  pre-filter and as the `Omega*` ground-truth for the coverage denominator.
- **`v_cert` (certificate, the real verifier):** the optimization above. Sufficient safety witness; returns `(P*,
  level sets)` = the "verified polytope" the spec wants for judging the FM output and for the less-conservative claim.

In the loop the label is `y = v_cert(U)` (which implies `v_collision`). We log when `v_collision=1` but `v_cert=0`
(collision-free yet *not certifiable* at `gamma_max`) — that delta is the residual conservativeness of the DTCBF
family and is itself a research signal (how non-conservative can `gamma` go before certificates vanish).

## 5. "candidate polytope <-> verifier" communication (spec)

`P_cand` (deterministic, from `polytope.py`) seeds the verifier's normal search (good warm-start angles) and defines
the conservative baseline. The verifier, by *finding* a different `(n_j, b)` that certifies an out-of-`P_cand`
trajectory, hands back an **updated, less-conservative polytope** `P*`. Over rounds the FM policy is trained on
sequences certified by these `P*` — so the policy ends up "respecting a NEW polytope", exactly spec (3). `P_cand` is
never the training target; it is only the seed/baseline.
