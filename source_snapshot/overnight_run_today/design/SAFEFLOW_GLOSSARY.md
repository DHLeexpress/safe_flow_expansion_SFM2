# SafeFlow Exploration — explicit definition of every term

This is the brainstorm the spec asked for: how each ACTFLOW term should be defined in **our** control/DTCBF
setting, written so it maps 1:1 onto code in `../src/`. The two most load-bearing terms — **Eq. (9)** (active
exploration) and **Eq. (10)** (uncertainty `sigma`) — get their own deep-dive section with an *iterative
self-diagnosis* protocol, because the whole method lives or dies on them.

Notation: a problem instance / context is `c = (x0, g, {obstacles})`. For ENV-A and ENV-B `c` is **fixed**
(overfit). A "design" is a control sequence; the dynamics are the 2-D double integrator.

---

## 0. The objects

### Design space `X`
```
U = (u_0, ..., u_{T-1}) ∈ R^{T×2},   u_t ∈ [u_min, u_max]^2   (acceleration, box-limited)
```
A design is a **whole control sequence** (spec 3.1: "FM 은 이제 delta U 가 아니라 sequences 자체를 배운다").
The rollout map is deterministic given `c`:
```
xi(U; x0):  x_{t+1} = A x_t + B u_t,   x=(p,v),   p_{t+1}=p_t+dt·v_t+0.5dt^2 u_t,  v_{t+1}=v_t+dt·u_t
```
We sometimes treat the design as `(U, c)`; for fixed-env experiments `c` is constant so `U` alone suffices.

### Valid (= safe) design space `Omega*`
The crux reinterpretation: **valid ≡ DTCBF-certifiable-safe**, *not* "matches the conservative corridor".
```
Omega*(c) = { U ∈ X :  v_cert(U, c) = 1 }
```
where `v_cert` is the verifier (below). This set is **strictly larger and less conservative** than the set of
sequences that stay inside the single deterministic `P_cand` from `polytope.py`. For ENV-A `Omega*` is **bimodal**
(left corridor ∪ right corridor); for ENV-B it is **trimodal** (left ∪ gap ∪ right). The conservative `P_cand`
typically captures only ONE leaf — that gap between `P_cand` and `Omega*` is exactly what Safe Flow Expansion fills.

### Generable set `Omega^tau_θ`
```
Omega^tau_θ(c) = { U : q_θ(U | c) ≥ tau }      # tau-level superlevel set of the FM policy density
```
Estimated, as in ACTFLOW's 2-D toy, by a **finite-sample histogram on a trajectory descriptor** (not on raw `U`,
which is too high-dim): sample `K` sequences, histogram their descriptors `d(U)` (see `METRICS.md`), threshold at
`tau` (paper used `tau=0.01`, 100×100 hist). This is how we make "coverage" measurable without one-shot generation.

### Verifier `v_cert : X × C -> {0,1}`  (spec 3.2: "verifier 은 optimization problem")
`v_cert(U,c)=1` iff there **exists** a polytope `P*` (separating half-spaces, one per obstacle) and a decay rate
`gamma ≤ gamma_max` such that the realized trajectory `xi(U;x0)`:
1. **containment**: stays on the safe side of every face of `P*` for all `t`;
2. **separation**: every obstacle (constant-velocity-predicted) stays on the far side with margin `r_j + r_robot`;
3. **DTCBF invariance**: `h_j(x_{t+1}) ≥ (1-gamma) h_j(x_t)` for every face `j`, every `t`, with the affine
   barrier `h_j(x)=[ (nearest0 - p)·n_j - eta (v - v_o)·n_j ] / d0b` (exact form from `safegpc_adapter/barrier.py`).

This is a small convex feasibility program (in 2-D: a sweep of LPs over the normal angle). When feasible it
**returns `P*` and its level sets** — the "verified polytope" of spec (1). Full formulation, complexity, and 3-D
scaling: `VERIFIER.md`. Cheap inner ground-truth (collision-free rollout) is a *necessary* pre-filter; the
certificate is the *sufficient* safety witness and the source of the less-conservative geometry.

### Pretrained policy `theta_0` (the seed)
Standard CFM on SafeMPPI / mirror-sampled SAFE sequences that respect the conservative `P_cand`. We **intentionally
restrict the seed to one homotopy leaf** (e.g. pass-right only) so the pretrained generable set is a narrow slice of
`Omega*` — mirroring ACTFLOW's misspecified pretrained mode (the `N((-1.1,0), 0.1^2)` blob in the chessboard).
The whole point is to watch Expansion grow it to the other modes.

### `phi_s` — noised-flow representation `Z_s`
`phi_s(U)` = the **penultimate hidden activation** of the FM velocity network evaluated on the *noised* sequence at
noise level `s ∈ (0,1)`:
```
U_s = (1-s)·noise + s·U          # CondOT interpolation; s≈0.9 = "close to data" (ACTFLOW 2-D used s=0.9)
phi_s(U) = h_{L-1}( v_theta(U_s, tau=s, c) )    # features before the final linear head
```
`Z_s` is this feature space. Rationale (paper): intermediate noise level gives a smoother, more globally connected
geometry than `s=1` (pure data), so uncertainty/acquisition generalizes across far-apart valid regions.

---

## 1. Eq. (10) — uncertainty `sigma_t`  (EXTRA CARE)

**Verbatim (paper):**
```
sigma_t^2(x) = k_{phi}(x,x) - k_{phi}(x,X_t) (K_{t,phi} + lambda I)^{-1} k_{phi}(X_t,x),    sigma_t = sqrt(.)
k_{phi}(x,x') = <phi_s^t(x), phi_s^t(x')>        # linear kernel on the learned representation
```
This is the **posterior variance of Bayesian linear regression** treating verifier labels `y=v(x)` as noisy linear
observations over `Z_s`. `X_t=(x_1..x_t)` = previously verifier-queried designs; `lambda` = obs-noise regularizer.

### Our definition (code: `src/uncertainty.py`)
- Buffer features `Phi = [phi_s(U_i)] ∈ R^{t×D}` for the queried `U_i`.
- **Linear kernel** (paper-faithful): `K = Phi Phi^T`, `k(x,X)=phi(x) Phi^T`, `k(x,x)=||phi(x)||^2`.
  Equivalent Woodbury feature-space form (cheaper when `D < t`, numerically nicer):
  ```
  sigma^2(x) = lambda · phi(x)^T (Phi^T Phi + lambda I_D)^{-1} phi(x)
  ```
  We use this form; `(Phi^T Phi + lambda I_D)` is `D×D`, refactorized once per round (Cholesky).
- **RBF kernel** (toy-faithful alternative, ACTFLOW 2-D used RBF on raw `x`, lengthscale `0.08`): selectable via
  config for sanity-checking against the paper's exact toy behavior.
- **Ensemble fallback** for high-D / many queries: 5 bootstrapped MLPs (2×100 ReLU, 10% dropout) predicting the
  verifier label; `sigma` = ensemble std. (Paper used this for molecules/proteins.) Kept as a config switch.

### Why each piece matters (and the failure modes)
- `lambda` too small → `K` ill-conditioned, `sigma` blows up / goes negative numerically. We clamp
  `sigma^2 ← max(sigma^2, 0)` and set `lambda` from the feature scale (`lambda = lam_rel · mean(diag(Phi^T Phi))`).
- **Representation drift.** `phi_s^t` changes every round because `theta` changes. So the kernel/buffer must be
  **recomputed from scratch each round** on the *current* `phi_s^t` (the superscript `t` in the paper is not
  cosmetic). Caching features across rounds is a silent bug.
- Linear kernel on a collapsed representation → `sigma` ~ flat → no useful acquisition. Diagnosis D6 below catches
  this; remedy is the RBF kernel or the ensemble.

### Self-diagnosis for Eq. (10) (logged to `results/diagnostics.json` each round)
- **D1 (cold start).** `t=0`, empty buffer ⇒ `sigma_0(x)=sqrt(k(x,x))`. RBF ⇒ constant `1` (std≈0). Linear ⇒ equals
  `||phi(x)||`. Assert the closed form matches a direct recompute. *Expectation: at t=0 the KL term dominates Eq.9,
  so we sample ~ the prior `q_{theta_0}` (exactly the paper's chessboard iteration-0 behavior).*
- **D2 (shrinkage at queried points).** For `U_i ∈ X_t`, `sigma_t(U_i)` should be ≈ `sqrt(lambda·...)` ≪ the prior
  `sqrt(k(U_i,U_i))`. Log `mean sigma @ buffer` vs `mean sigma @ fresh`.
- **D3 (monotonicity / information never hurts).** Adding a query must not raise `sigma` anywhere:
  `sigma_{t+1}(x) ≤ sigma_t(x)` (for fixed `phi`). Spot-check on a held grid; flag violations (would indicate a
  kernel/Woodbury sign bug).
- **D4 (novelty correlation).** `sigma_t(x)` should rise with distance-in-`phi_s` from the buffer. Log Spearman
  corr between `sigma` and `min_i ||phi(x)-phi(U_i)||`; expect strongly positive.
- **D6 (representation collapse guard).** Log `rank`/`effective dim` of `Phi` and `std(sigma)/mean(sigma)` over a
  fresh batch; near-zero variance ⇒ collapsed `phi_s` ⇒ switch kernel.

---

## 2. Eq. (9) — active exploration  (EXTRA CARE)

**Verbatim (paper):**
```
x_{t+1} ~ p~_t ∈ argmax_q  E_{x~q}[ sigma_t(phi_s^t(x)) ]  -  beta · KL( q || p_1^{theta_t} )
```
- 1st term: favor informative queries about the verifier `v` (and thus about `Omega*`).
- KL term: regularize toward the current generative prior so exploration stays where validity bias is useful.
- `beta`: exploration↔prior trade-off. `beta→∞` ⇒ plain sampling `x~p_θ`; `beta→0` ⇒ pure uncertainty max (can
  wander to regions the prior no longer informs). Toy used `beta=1/13≈0.077`.

### How it is actually solved (this is the subtle part)
The paper does **not** run gradient ascent or an explicit optimizer over `q`. The unconstrained maximizer of
`E_q[sigma] - beta·KL(q||p)` is the **Gibbs / exponential tilt** of the prior:
```
q*(x) ∝ p_1^{theta_t}(x) · exp( sigma_t(phi_s^t(x)) / beta )
```
So in practice it is an **inference-time uncertainty-tilted sampling oracle**: draw a batch from the *current FM
policy*, then reweight by `exp(sigma/beta)`. The KL term is honored *for free* because we sample FROM the policy and
only tilt — we never leave the policy's support.

### Our definition (code: `src/safeflow.py::active_exploration`)
1. Sample `N` candidate sequences from the current policy: `U^{(1..N)} ~ q_theta(·|c)` (Euler ODE from noise).
2. Compute `sigma_t(U^{(i)})` via Eq. (10) on `phi_s^t`.
3. Tilt weights `w_i ∝ exp(sigma_t(U^{(i)}) / beta)`; **systematic-resample** (or top-`B`) to pick `B` queries.
   - Numerical: subtract `max sigma` before `exp`; cap `sigma/beta` to avoid overflow.
   - `beta` schedule option: anneal `beta` down over rounds (more exploitative early, more exploratory later) — off
     by default to stay paper-faithful.
4. Query the verifier on the `B` selected sequences → labels `y`.

> Why this beats naive "sample-and-check": tilting spends the verifier budget on sequences near the *boundary of the
> current generable set* (high `sigma`) — precisely the local-to-global crawl that lets the chessboard jump cells.
> Pure `p_theta` sampling (`beta→∞`) re-confirms the known mode and never crosses into the other homotopy leaf.

### Self-diagnosis for Eq. (9)
- **D7 (tilt sanity).** At `t=0` with flat `sigma`, the resample must reduce to i.i.d. policy sampling (weights ≈
  uniform). Log effective sample size `ESS = (Σw)^2/Σw^2`; at t=0 `ESS≈N`.
- **D8 (boundary targeting).** Log the descriptor histogram of *selected* queries vs *all* policy samples; selected
  should shift toward un-covered descriptor bins as rounds progress.
- **D9 (verifier yield).** Log fraction of selected queries that come back SAFE. Healthy expansion: starts moderate,
  rises as the policy learns the new leaves; a collapse to ~0 means `beta` too small (wandering OOD).

---

## 3. UpdateFlow — `theta_{t+1} = UpdateFlow(theta_t, D_{t+1})`

Partition the buffer into `D^+` (verifier-SAFE) and `D^-` (UNSAFE). CFM (Cond-OT) loss:
```
g_t = grad L^+(theta) - alpha_t · grad L^-(theta)            # signed gradient: learn safe, unlearn unsafe
L^±(theta) = E_{U~D^±, noise, tau} || v_theta(U_tau, tau, c) - (U - noise) ||^2
```
- `alpha_t` = strength of the negative/unlearning signal. Toy used `0.005`; many ACTFLOW runs used `0`
  (positive-only). We default `alpha_t=0.005`, with the paper's online calibration
  `alpha_t = alpha · ||grad L^+|| / ||grad L^-||` available as a switch.
- **Warm-up.** No FM update until `N_warmup` SAFE samples are collected (paper deferred fine-tuning until 4096 valid;
  we scale down, e.g. `N_warmup ≈ 1–2 batches`), so `sigma` is trustworthy before we start moving `theta`.
- This is step **(2)→(3)** of the spec: bulk SAFE samples → new policy → respects the **new** (less conservative)
  polytope geometry, because the SAFE set it now fits is `Omega*`, not `P_cand`.

---

## 4. The full loop (code: `src/safeflow.py::run_safeflow`)

```
pretrain theta_0 on conservative SafeMPPI seed (one leaf)
D_0 = ∅
for t = 0..T-1:
    re-extract phi_s^t from theta_t                          # representation co-evolves (do NOT cache)
    fit sigma_t from D_t over phi_s^t                        # Eq.10
    U_query = active_exploration(theta_t, sigma_t, beta, N, B) # Eq.9 (tilted resample)
    y = verifier(U_query, c)                                 # certificate optimization (VERIFIER.md)
    D_{t+1} = D_t ∪ {(U_query, y)}
    if |D^+| ≥ N_warmup: theta_{t+1} = UpdateFlow(theta_t, D_{t+1})   # else theta unchanged
    if t % eval_every == 0: log coverage, validity, Vendi, D1..D9   # METRICS.md + diagnostics
return theta_T
```

The three quantities that **co-evolve** (paper's key point): `theta_t` (policy), `phi_s^t` (representation, same
net), `sigma_t` (uncertainty, refit on the growing buffer). Treat them as one coupled system; the diagnostics above
are how we keep that coupling honest, iteratively, every round.
