# HP100 SFM port of the 3-D ball expansion protocol

Status: **frozen overnight sweep ready**.  The fixed-radius 2-D certificate,
candidate-specific geometry, and bounded parallel gathering smoke have passed
their visual and numerical approval gates.

## Intended invariant

The port changes the task adapter, not the expansion mechanism:

- the policy is the promoted SFM HP100 checkpoint;
- gathering uses the declared severe OOD profile: 40 pedestrians with desired
  speeds in 1.0--2.0 m/s;
- only the final flow head is trainable (5,140 of 1,978,068 parameters,
  approximately 0.26%);
- every active context samples `K=16` flow plans, uncertainty-selects `B=4`,
  and verifies those four plans independently;
- an episode executes one full-H exact-positive plan and replans, or terminates
  as NVP when `B` contains no full-H exact-positive plan;
- there is no controller, guidance, raw-plan, or expert fallback;
- exactly one batch of 16 episodes is advanced synchronously for each gamma and
  round; NVP lineages terminate independently and are never success-retried;
- every resolved selected-`B` full-`H=10` query is archived through each
  lineage's NVP, success, collision, or timeout decision: exact positives form
  `D+`, exact negatives form `D-`, and neither terminal status nor query
  multiplicity changes a row's label;
- training and evaluation scenario banks are disjoint.

The vendored core's native checkpoint uses a task-agnostic config payload.  At
delivery, every round is therefore exported again with the authenticated HP100
architecture contract so the existing canonical raw evaluator can load it
without tensor-shape inference.  Raw evaluation remains temperature one, with
no acquisition, verifier selection, guidance, or fallback, over disjoint OOD
pedestrian scenario IDs.

This archive is not terminal-success-conditioned.  Failed and NVP lineages
contribute all resolved preterminal queries: their exact positives enter the
ordinary positive objective, while exact negatives enter only arms with
nonzero signed-gradient strength.  It is still acquisition-conditioned because
only the uncertainty-selected `B=4` of `K=16` plans are labeled and archived.
Per-gamma first-action covariance and route/side entropy therefore remain
required diagnostics rather than assumed multimodality guarantees.

For SFM, the 16 replicas must be 16 different pedestrian scenarios.  The ball
implementation's replicas differ only through flow noise because its scene is
fixed; copying that reset logic would not provide SFM environment coverage.
Scenario IDs are paired across gamma: the same `(round,retry_batch,replica)`
uses the same pedestrian realization for every gamma, while different replicas,
retry batches, and rounds use different realizations.
The canonical expansion range starts at 300000.  It is disjoint from HP100
pretraining episodes (approximately 0--529), matched-ID evaluation
(`150000:150049`), and severe-OOD evaluation (`250000:250049`).
For the mechanism figure only, a diagnostic mode may replay 16 independent
noise lineages in one fixed pedestrian scenario so the branching effect is
visually comparable.  Such a fixed-scenario trace is audit-only and must not
be described as the production gathering distribution.

The reference adaptive-beta implementation fits one global beta from pooled
context/gamma score pools.  The port preserves that behavior and targets
normalized ESS `0.1`, starting from the ball runner's no-manifest fallback
`beta=5e-4`, but it must report realized ESS separately for every
gamma so a pooled calibration cannot hide a gamma-specific uniform arm.
The RBF length scale is initialized from exactly 50 authenticated HP100
pretraining contexts, using the original globally disjoint pretraining split.
Those contexts are gamma-balanced (one gamma contributes eight and each other
gamma seven), use distinct successful training lineages within each gamma,
and reconstruct the actual `[10,32,100]` history, `low5`, and control history.
One temperature-one pretrained plan and its original flow base are sampled per
context and embedded with paired noised `phi_s` at `s=0.9`; OOD expansion
states are not used for this calibration.
Because the HP100 encoder, GRU, low-state encoder, and trunk/phi are frozen,
`sliding_positive_per_gamma_frozen_phi` is the explicit reference in this
head-only arm. Candidate locations change as the output head changes; the
feature map itself does not drift.
The RBF support is a bounded sliding coreset over all strictly earlier rounds,
not just the immediately preceding round. It retains older exact-positive rows
when newer rows do not fill a gamma quota, while never reading the round being
gathered. The `trajectory_uniform` selector first enforces equal gamma capacity
and then near-equal lineage and departure-to-tail time-stage coverage.

`parallel_episodes` is a synchronous lineage scheduler, not a guarantee that
all policy forward calls are fused into one GPU tensor: the current reference
samples `K` plans one active context at a time, then batches the verifier work.
The first faithful port preserves this ordering.  A later vectorized sampler
must pass an equivalence test before it is described as the same algorithm.

## Exact 2-D GREEN certificate

For a candidate robot window and a currently sensed pedestrian disk, let

\[
r_t=p_t-p_0,\qquad d_j=p^{\rm ped}_{j,0}-p_0,\qquad
\beta_t=1-(1-\gamma)^t.
\]

For each current disk and each of 16 artificial boundary disks, the paper
block solves

\[
\max_{a_j,m_j}m_j
\]

subject to

\[
a_j^\top r_t\le\beta_t m_j,\quad
R_j\lVert a_j\rVert\le a_j^\top d_j-m_j,\quad
\lVert a_j\rVert\le1,\quad m_j\ge m_{\min}.
\]

The implementation solves this two-dimensional angular feasibility/max-margin
problem analytically; it does not discretize the normal angle. The radius is
the fixed finite `R_sense=2 m`; candidate displacement never enlarges it. A
positive max-margin solution has `||a_j||=1` and
`m_j=a_j^T d_j-R_j`, so every solved face is obstacle-tangent. The nominal
current-tangent polytope is one feasible SOCP point whenever its contraction
gate holds; the optimized verifier is therefore no more conservative than
that nominal choice. A label is positive only when the shared clipped rollout
is in-bounds, collision-free under the separately predicted moving pedestrians,
and GREEN-certified under the current-sensed paper geometry.

## Execution score and lambda preflight

Among exact-positive candidates, the requested selector is

\[
j^*=\arg\min_j\left[J_{\rm SafeMPPI}(U^j)-\lambda m_{\rm step}^j\right],
\]

where

\[
m_{\rm step}^j
=H_P^{\rm nominal}(p_{t+1}^j)
-(1-\gamma)H_P^{\rm nominal}(p_t).
\]

`m_step` is the BLUE current-tangent nominal-polytope contraction margin.  It
is not GREEN verifier slack.  The 3-D value `lambda=70000` is task-specific:
its native cost and dimensionless margin have different scales from SFM.

The absolute margin is gamma-dependent through
`(1-gamma) H_P(p_t)`, and gamma also changes the conditioned proposals and the
exact-positive set. For candidates sharing one fixed context and gamma,

\[
J_j-\lambda m_{\rm step}^j
=J_j-\lambda H_P(p_{t+1}^j)
+\lambda(1-\gamma)H_P(p_t),
\]

the final term is common and cancels in pairwise score differences. We retain
the complete gamma-dependent definition and do not silently add a second
`lambda_gamma` normalization.

Before training, collect native-cost and margin spans at exact-positive `B`
sets containing at least two candidates.  Define the diagnostic scale

\[
\lambda_0=
\frac{\operatorname{median}(\max J-\min J)}
     {\operatorname{median}(\max m_{\rm step}-\min m_{\rm step})}.
\]

Screen multipliers `{0, 0.5, 1, 2, 4}` without updating the model, and report
selection agreement with pure cost and pure margin, per-gamma NVP/success,
clearance, and time.  This is unit calibration, not post-hoc metric tuning.
The production launcher therefore has no implicit lambda: `run` mode fails
closed unless `--execution-step-margin-weight` is supplied explicitly.

## Deliberate deviations from the supplied ball command

1. `--inner-steps 10` in the ball CLI means ten repeats of every replay
   microbatch, not ten optimizer steps total and not one exposure per sample.
   The SFM launcher must name this parameter `microbatch_repeats` explicitly.
2. The SFM preterminal archive stores each selected-B full plan together with
   its original sampled base. `--paired-noised-representation` is canonical:
   every `(x0,U)` is authoritative and no bases from different replans are
   stitched together.
3. The supplied ball command's `committed_success` event logging removes
   failed/NVP traces.  This SFM port instead records and archives the bounded
   preterminal history of every lineage; the approval smoke likewise retains
   all 16 lineages.
4. The ball task adapter adds a strict progress gate after the exact
   certificate.  The requested SFM definition does not: every resolved full-H
   exact positive is execution-eligible, and progress/nominal margin are
   scores and diagnostics only.  Thus SFM NVP means exactly “no exact-positive
   member of B.”
5. The ball archive can label a terminal prefix with horizon shorter than ten
   and zero-pad it for CFM.  SFM rejects those prefixes: reaching the goal ends
   the episode, but never weakens the declared full-`H=10` GREEN label.

## Approval smoke artifacts

With `sliding_positive_per_gamma_frozen_phi`, round 1 has an empty GP support;
all prior variances are equal and a low ESS target cannot create meaningful
uncertainty contrast.  Therefore the mechanism approval should show both the
uniform round-1 bootstrap and one round-2 batch after round-1 successful
support has populated the GP.  Only the latter is evidence for uncertainty
tilting.

For one round, one gamma, and one retry batch, retain the identity

```text
(round, gamma, retry_batch, replica, lineage_id, step)
```

and render:

- a 4-by-4 synchronized lineage video (`K` gray, selected `B` orange, exact
  positives green, exact negatives red, executed step blue, committed success
  gold, and per-lineage NVP red X);
- a candidate-specific `B=4` snapshot using the exact stored GREEN faces and
  ten level sets;
- a terminal summary showing every attempt and its terminal decision;
- a hash manifest containing source, checkpoint, scene, dynamics, solver, and
  trace identities.

Rendering must consume stored solver outputs and must never rerun or replace
the verifier.

The bounded local approval smoke at fixed severe-OOD scene `250007`,
`gamma=0.5`, used 16 independent flow-noise lineages, `K=16`, `B=4`, and no
fallback.  It produced 85 contexts, 1,360 generated plans, 340 exact queries
(222 positive, 118 negative), and 71 executed first actions.  Fourteen lineages
reached a context with no exact-positive member of `B`; two reached the bounded
diagnostic cutoff.  This is an intentionally negative diagnostic, not a
committed training batch.  Its one-gamma unit calibration gave
`lambda_0=135,565` and `0.5 lambda_0=67,783`, placing 70,000 near a half-scale
blend for this smoke only.

Round 1 has no GP support, so marginal sigma is uniform regardless of the low
beta. The current smoke therefore validates branching, exact labels, and
fail-closed execution, but it is not evidence that ESS 0.1 uncertainty tilting
works. Production archives every resolved selected-B query through each
lineage's terminal decision. Exact positives enter D+ and the next round's
gamma/lineage/time-balanced GP; exact negatives remain D- for the declared
signed-alpha arms and never enter the GP.
