# SFM Hp10 + B1: corrected ID/OOD protocol

## What changed after `103476d`

The original result files did not serialize their crowd environment.  A source
audit established that the `103476d` expansion, M20 screening, and M100
confirmation all used 20 pedestrians at 1.0--1.5 m/s.  They were therefore a
velocity-OOD test relative to the training data (20 pedestrians at 0.5--1.0
m/s), not the requested ID test.

Every new command is fail-closed on an explicit scene profile:

| profile | pedestrians | speed [m/s] | interpretation |
|---|---:|---:|---|
| `training` | 20 | 0.5--1.0 | demonstration distribution |
| `id` | 10 | 0.5--1.0 | requested easier benchmark; lower density than training |
| `legacy_velocity_ood` | 20 | 1.0--1.5 | exact `103476d` environment |
| `requested_ood` | 30 | 1.0--1.5 | density-plus-velocity OOD |
| `density_ood` | 50 | 0.5--1.0 | density-only OOD at the training velocity range |

This distinction is stored in every new metrics JSON.

The current density-only study uses `density_ood`: it changes only pedestrian
count from the demonstration distribution (20 to 50) while holding the sampled
speed range fixed at 0.5--1.0 m/s. Its fixed M100/gamma deployment bank begins
at scenario 210000 and is declared before observing outcomes.

## Model and expansion semantics

The Hp10 policy uses the ten most recent nominal-polytope (H_P) grids and a
frozen visual encoder.  At each alive context, B1:

1. draws (K=16) windows from the raw conditional flow;
2. uses an RBF-GP posterior uncertainty to query (B=4) windows;
3. stores every resolved query in (D), and every full-horizon positive in
   (D^+);
4. executes only a resolved queried plan with \(y=1\) and
   (H_P(x_{t+1})\ge(1-\gamma)H_P(x_t)); every \(y\) is computed from the
   complete H=10 window, even when its predicted states cross the goal;
5. terminates that replica on no verified positive (NVP);
6. replays the most recent (W=2) rounds with equal mass over
   gamma -> (round, episode) -> context -> positive query.

Arm A ranks admissible queries by maximum one-step nominal margin.  Arms B--D
rank the same admissible queried sequences by the frozen SafeMPPI proposal cost:

\[
j^\star=\arg\min_{j:\,y_j=1,\,
H_P(x_{t+1}^j)\ge(1-\gamma)H_P(x_t)} J_{\rm SafeMPPI}(U^j).
\]

Thus the cost never admits an unverified plan and never generates an expert
proposal.  B, C, and D differ only in negative-gradient coefficient
\(\alpha\in\{0,10^{-3},10^{-2}\}\).

The authenticated preflight uses exactly 50 gamma/scenario-balanced pretrained
embeddings:

- \(\ell_0=0.48421653441442203\)
- selected multiplier (0.5), \(\ell=0.24210826720721101\)
- original preflight-selected cap 256, \(\lambda=0.01\)
- calibrated \(\beta=0.11756989408559083\), normalized ESS 0.5
- uplift 0.06455287337303162, condition 540.5854, effective rank 30.6369

The current alpha-by-replay-epoch study keeps this authenticated length scale
but uses the stable cap-512 preflight row. After each round it retains at most
256 upper-quartile full-H positives, using rotating per-gamma quotas of 36/37;
the two-round GP memory therefore contains at most 512 records with 73/74 per
gamma when both rounds fill. No quota is borrowed across gamma. The GP remains
fixed across all 56 episodes inside a macro-round, and beta is recalibrated once
at the start of every round to normalized ESS 0.5. These are acquisition
contracts, not a claim that every later sequential selection has exactly the
same realized ESS.

The current replay sweep fixes 16 optimizer chunks, learning rate \(10^{-4}\),
and varies \(\alpha\in\{0,10^{-3},10^{-2}\}\) and complete replay epochs
\(E\in\{1,4,16\}\). Every eligible record in W=2 is visited exactly E times.
The original gamma -> (round, episode) -> context -> query mass is retained
across the 16 chunks rather than being renormalized independently per chunk.

On Helios, the sweep scheduler exposes eight process slots: `1a`--`1d` on
physical GPU 1 and `3a`--`3d` on physical GPU 3. Thus four independent arm
processes share each GPU; every process receives 8 disjoint verifier CPU
workers. Before any scientific sweep, a fail-closed three-round runtime gate
runs eight representative alpha/epoch configurations under this exact scheduler. The
gate authenticates the source, checkpoint, preflight, scene, GPU assignment,
GP cap/quota behavior, and measured wall-time forecast before later jobs are
admitted.

## Honest selection of the existing checkpoint

Arm A round 10 remains the selected checkpoint because it won the declared,
frozen M20 screening rule.  D looked slightly better only on the later disjoint
M100 confirmation.  Replacing A by D after reading confirmation would leak the
confirmation bank.  D can be reported as a post-hoc diagnostic, not relabeled as
the selected method.

## Evaluation contract

Raw evaluation means temperature one, NFE 8, one generated window per context,
and execution of its first action.  It uses no RBF uncertainty, verifier,
SafeMPPI selector, or fallback.  Kazuki is separately named because it adds goal
and safety guidance plus MPPI refinement.

The current alpha-by-replay-epoch study uses the following leak-separated
selection sequence:

1. Every arm receives canonical raw temperature-one M10/gamma evaluation at
   every round. These records are development monitoring, not confirmation.
2. The development records nominate four checkpoints: the best checkpoint for
   each of the three alpha values, plus the best distinct remaining candidate.
3. For each shortlisted checkpoint, a disjoint M10/gamma bank selects and locks
   one seven-gamma temperature vector. Only that locked vector receives one
   M50/gamma screening evaluation.
4. The winning arm, round, and temperature vector are frozen from the M50
   screen. An untouched M100/gamma bank then performs the paired final
   confirmation: canonical temperature one and the locked temperature vector.

No exhaustive post-hoc rerun of every checkpoint is part of this protocol.
Every Helios artifact produced by the study--including logs, checkpoints,
metrics, manifests, figures, and videos--must be rooted under
`/data3/research1`; the source worktree contains code only.

`sfm_b1_deploy_driver.py` performs:

- matched M100/gamma deployment of Hp10 r0, selected A-r10 raw, and default
  Kazuki on `id` and `requested_ood`;
- raw M50/gamma evaluation of every A checkpoint, rounds 0--20, on fixed banks;
- separate ID/OOD gallery PNG/MP4 with radius-0.2 pedestrian disks;
- a diagnostic-only paired OOD rollout for the margin and SafeMPPI-cost
  selectors, followed by two fixed-frame 3x3 certificate figures.

The query figure colors are fixed: gray (K), orange queried (B), green complete
H=10 verifier-positive, red rejected, and thick blue executed first action.
Blue nominal and green verifier
polytopes contain exactly the ten gamma-dependent horizon sets.  At
\(\gamma=1\), the ten sets genuinely coincide and are labeled as such.

The moving-pedestrian verifier used here is the compact 2-D implementation in
`sfm_metrics2.py`: a 180-angle moving-face fit followed by a direct certificate
check.  It evaluates the intended polytope conditions, but it is not a generic
CVXPY/conic-solver call.  Consequently, measured verifier latency is reported
as angular-fit certificate latency, not as real-time SOCP-solver latency.  A
returned witness is checked against the complete window and is therefore sound
under the constant-velocity pedestrian model; discretizing the face normal can
miss a feasible continuous-normal witness, so the implementation is not a
complete SOCP feasibility oracle.

The paired query diagnostic never enters (D,D^+), the GP, or model training.
It chooses encounter snapshots by a declared minimum-distance rule over both
selectors, rather than visual curation.

For the density-only study, `sfm_b1_density_diagnostic.py` keeps two evidence
streams separate:

- the unbiased M100/gamma deployment bank starts at scenario 210000;
- the finite visualization-search bank starts at scenario 230000 and is
  explicitly outcome-conditioned explanatory evidence, never an evaluation.

The finite search evaluates every declared episode before applying its fixed
tier rule. It then reruns the chosen episode with traces enabled and verifies
that all nine method/gamma outcomes match the search. The 3x3 comparison uses
one shared step selected by minimum mean robot--pedestrian distance over all
nine cells. The max-margin query snapshot uses the fixed composition order
P3/N1, P2/N2, then P4/N0 and breaks ties by normalized full-window control
spread. No visual inspection changes either choice.

`sfm_b1_density_viz.py` is render-only. Hp10 r0 and selected A-r10 are raw
temperature-one rollouts: the blue nominal set and green selected-window
certificate are diagnostic overlays and did not select either raw action.
Kazuki is shown separately with its accumulated guidance vector. In the
max-margin gathering video, no nominal set is drawn and only the actually
executed full-H positive query may own a green verifier polytope. Each panel of
the B-query snapshot instead fits its own candidate-specific verifier; rejected
queries never receive a green H=10 set. Reaching the goal terminates the real
closed-loop episode only after the executed first action; it never truncates a
candidate certificate or creates an absorbing terminal prefix.

## Corrected deployment result (`b2caf9a_id_ood_deploy`)

The fixed M100/gamma benchmark uses 700 rollouts per method.  The selected
checkpoint remains arm A, round 10; these measurements did not reselect it.

| profile | method | SR | CR | successful clearance [m] | successful time [s] |
|---|---|---:|---:|---:|---:|
| `id` | Hp10 r0 raw | 0.9800 | 0.0200 | 0.4944 | 5.3125 |
| `id` | selected A-r10 raw | 0.9786 | 0.0214 | 0.4968 | 5.3835 |
| `id` | default Kazuki | 0.9986 | 0.0014 | 0.4437 | 3.1568 |
| `requested_ood` | Hp10 r0 raw | 0.8457 | 0.1514 | 0.1611 | 7.6608 |
| `requested_ood` | selected A-r10 raw | 0.8486 | 0.1500 | 0.1603 | 7.7721 |
| `requested_ood` | default Kazuki | 0.9171 | 0.0814 | 0.2278 | 4.0431 |

The lower-density requested ID profile is already saturated, and expansion does
not improve it.  On density-plus-velocity OOD, A-r10 changes SR by only
0.29 percentage points and CR by only 0.14 percentage points.  Default Kazuki
is stronger but still misses the requested empirical CR below 5%.

The fixed raw M50/gamma checkpoint curve on `requested_ood` is also modest and
non-monotone: pooled SR is 0.8429 at r0, 0.8514 at r10, peaks at 0.8714 at r19,
and ends at 0.8629 at r20.  The r19 value is a diagnostic observation, not a
post-hoc replacement for the frozen A-r10 selection.

Finally, all 18 diagnostic-only closed-loop runs (three scenarios, three gamma
values, and two selectors) terminate by NVP after 13--23 steps.  SafeMPPI cost
can rank an already-admissible queried set, but it does not repair loss of
finite-K/B support.  The paired 3x3 figures retain these NVP cells rather than
curating successful snapshots.

The authenticated output is stored at
`/home/dohyun/projects/sfm_hp10_b1_runs/b2caf9a_id_ood_deploy`; its delivery
manifest contains 113 independently rehashed artifacts.

## Density-only pre-expansion result (`density_ood_deploy_sharded_927d1a3`)

The requested density-only profile keeps the configured desired-speed range at
0.5--1.0 m/s and changes `n_ped` from 20 to 50.  A process-isolated fixed bank
evaluated 100 predeclared scenario IDs per gamma (700 rollouts per method):

| method | SR | CR | timeout | successful clearance [m] | successful time [s] |
|---|---:|---:|---:|---:|---:|
| Hp10 r0 raw | 0.9957 | 0.0043 | 0.0000 | 0.2064 | 9.5900 |
| selected A-r10 raw | 0.9914 | 0.0043 | 0.0043 | 0.2075 | 9.6591 |
| default Kazuki | 0.9900 | 0.0100 | 0.0000 | 0.2330 | 4.8036 |

The pooled intervals resample scenario IDs while retaining all seven gamma
rows.  All 21 method--gamma cells and output hashes passed authentication.  The
sharded run took 16 minutes 53 seconds on physical GPU 3; the unchanged serial
launcher would have taken about four hours because SFM pedestrian interaction
is CPU-serial and quadratic in pedestrian count.

This result rejects density-only 20-to-50 as a discriminative expansion target:
r0 already satisfies the requested empirical CR below 5%, and A-r10 does not
improve it.  No new expansion was launched from this pre-expansion gate.

## Known blind spots

- A fitted H=10 verifier proves the queried window, not infinite-horizon
  viability under moving pedestrians.
- The current compact verifier uses an `n_theta=180` angular moving-face fit
  followed by a direct certificate check. It is not a generic CVXPY solve;
  measured executor wall time per queried candidate must accompany real-time
  claims.
- The constant-velocity pedestrian prediction is a modeling assumption.
- The requested `id` profile has lower density than the demonstration data and
  is therefore an easier matched-speed benchmark, not literally the full
  training distribution.
- The RBF embedding can still become insensitive to behaviorally different
  windows; uncertainty uplift and realized ESS must be monitored.
- One gallery episode is explanatory evidence only.  Claims must use fixed-bank
  M100 metrics and confidence intervals.
- `goal_coef=safe_coef=0` does not make Kazuki identical to raw flow because its
  MPPI refinement and warm start remain active.

## Entry points

- `sfm_scene.py`: scene and profile contracts
- `sfm_hp_history.py`: Hp10 construction
- `grid_policy_sfm.py`: flow policy and frozen visual encoder
- `sfm_metrics2.py`: full-H moving-pedestrian verifier
- `sfm_b1_rbf.py`: RBF-GP uncertainty and adaptive ESS calibration
- `sfm_b1_store.py`: round-sharded (D,D^+,D^-) and exact hierarchical replay
- `sfm_b1_cost.py`: one-step (H_P) gate and execution selectors
- `sfm_b1_expand.py`: B1 gathering and update
- `sfm_b1_eval.py`: raw evaluation
- `sfm_b1_benchmark.py`: matched deployment and checkpoint curves
- `sfm_b1_sharded_benchmark.py`: process-isolated 3-method x 7-gamma fixed
  M100 deployment with exact cell authentication and episode-cluster pooled CIs
- `sfm_b1_query_diagnostic.py`: paired closed-loop selector diagnostics
- `sfm_b1_viz.py`: gallery, query paths, and ten-level polytopes
- `sfm_b1_density_diagnostic.py`: disjoint finite case search, traced method
  reruns, max-margin query collection, and verifier timing
- `sfm_b1_density_viz.py`: render-only 3x3, gathering MP4, and candidate-specific
  query snapshot
- `sfm_b1_deploy_driver.py`: authenticated two-GPU delivery driver
