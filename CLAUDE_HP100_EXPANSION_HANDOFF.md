# Claude handoff: HP100 always-on predictive Safe Flow Expansion

## 0. Read this first

`DHLeexpress/safe_flow_expansion_SFM2` is a **public** source repository. Work
in a private local checkout and a private development branch; do not modify a
shared checkout or reuse another agent's output root.

```bash
cd /home/dohyun/projects
git clone https://github.com/DHLeexpress/safe_flow_expansion_SFM2.git \
  safe_flow_expansion_SFM2-claude
cd safe_flow_expansion_SFM2-claude
git fetch origin
FROZEN_SHA="$(git rev-parse origin/main)"
git switch -c agent/claude-sfm2-predictive-$(date +%Y%m%d) "$FROZEN_SHA"
python scripts/show_claude_hp100_handoff.py
python scripts/verify_package.py
```

Show the frozen SHA and complete checker output before editing. Use a new
Helios output root:

```text
/data3/research1/claude_sfm2_predictive_<FROZEN_SHA7>/
```

Generated checkpoints, tensors, logs, and evaluation banks remain outside
Git. Commit only source, tests, compact JSON/CSV provenance, and explicitly
requested figures/videos to your branch.

## 1. Scientific status and task

The canonical pretrained checkpoint and the no-update always-on acquisition
diagnostic are available. A complete multi-round updater and an independent
evaluation funnel still need to be implemented. **No long run has validated
that this protocol improves performance.** Do not describe the mechanism as a
successful expansion result until fresh fixed-bank evaluation supports it.

Your task is to implement the positive-minus-alpha-negative update around the
frozen acquisition contract, qualify it for one round, then search for a fixed
recipe that lowers OOD collision rate and raises window Validity and successful
clearance without destroying SR or time to goal.

## 2. Immutable checkpoint, data, scene, and verifier

| Artifact | Path | SHA-256 |
|---|---|---|
| Canonical checkpoint | `checkpoints/hp100_pretrained_r0_258999ae.pt` | `258999ae8ccee8aec5aab92a6f751221d3c15583ac26e0a7ec8311f13316ec44` |
| Pretraining report | `provenance/hp100_pretrain_20260802/pretraining_report.json` | `19f83f865056db73aea656d617680c1078ed5cf69e0a5fdb68cc4f5c14b74cbd` |
| Dataset manifest | `provenance/hp100_pretrain_20260802/dataset_manifest.json` | `44f2bfa8afbb2318376ae9e188b1b622f102253a4a91f5c8ca0f9634d5041c94` |
| ID raw M50 | `provenance/hp100_pretrain_20260802/id_m50.json` | `58df71d0a25b801c47f9f0da8077704eddb0b52eda18c072de45cad2c3818961` |
| OOD raw M50 | `provenance/hp100_pretrain_20260802/ood_m50.json` | `1708348be707868d93e8e878f069cb66c0605fea33ab1f4319dd3d3cf1b0ce4e` |

External Helios data:

```text
/data3/research1/sfm_hp100_certified_weighted_500x7_2671a94
```

Authenticate it through `DATA_POINTER.json`; do not regenerate or relabel the
pretraining dataset.

Locked contracts:

- checkpoint state/config and canonical raw r0 results;
- ID: 20 pedestrians at 0.5-1.0 m/s;
- OOD: 40 pedestrians at 1.0-2.0 m/s;
- start `(0,0)`, goal `(6,6)`, `dt=.1`, action and velocity caps `[-2,2]`;
- gammas `{.1,.2,.3,.4,.5,.7,1.0}`;
- current-tangent HP100 observation, `predict_gain=0`, 16 nominal outer faces;
- candidate-specific exact full-H10 GREEN verifier, clipped dynamics, all-
  pedestrian CV collision check, analytic H1-H10 faces, 16 artificial faces;
- collision, success, timeout, clearance, and window-Validity definitions;
- raw temperature-one, NFE-8 evaluator and its CRN contracts.

The learned condition remains the 10x32x100 Hp history, low state including
gamma, and GRU control history. Raw pedestrian positions and velocities are
available only to the verifier/acquisition selector. Do not add them to the
policy input.

## 3. Authoritative acquisition contract

Start from
`source_snapshot/overnight_run_07_12_sfm/sfm_hp100_predictive_execution.py`.
It is a no-update, default-off evidence collector; extend it additively rather
than changing baseline/evaluator semantics.

At every replan:

1. Generate `K=64` H10 flow proposals and retain their original Gaussian
   bases.
2. Embed the paired-noised proposal using the frozen reference representation
   at `s=.9`.
3. Acquire `B=32` without replacement with the RBF posterior and adaptive beta
   at normalized `ESS/K=.1`.
4. Resolve all B using the exact full-H10 GREEN verifier at that context's
   gamma.
5. Among exact-positive rows select lexicographically by:

   ```text
   maximum H10 goal progress
   maximum CV-predicted minimum clearance
   maximum selected uncertainty
   minimum candidate index
   ```

   Equivalently,

   \[
   j^\star=\operatorname*{arg\,max}^{\rm lex}_{j:y_j^\gamma=1}
   (\Delta_g^j,c_{\rm CV}^j,\sigma_j,-j).
   \]

6. Execute only the first clipped action of the selected exact-positive H10
   proposal, then refresh the real SFM state. If that immediate transition
   realizes collision/OOB, archive that executed row as a realized negative,
   preserve its original verifier `y=1` separately, and terminate.
7. If B has no exact positive, hold state fixed and retry with Gaussian
   flow-base std `1.0, 1.1, ...` for 32 attempts (`a=0,...,31`) in the
   authoritative diagnostic. This is not ODE temperature tuning.
8. If the final B is still all negative, execute none of them. Archive exactly
   one resolved negative counterfactual, selected by H10 progress, then CV
   clearance, then index, and terminate that lineage NVP.

Exact-positive eligibility already contains the constant-velocity collision
test. CV clearance is an assertion/ranking statistic, not a replacement for
the SOCP label and not an additional permissive safety gate.

## 4. Archive and objective

Use only these roles:

- `D+`: each selected, executed, exact full-H10 positive whose immediate live
  transition does not realize collision/OOB;
- `D- / all_negative_nvp`: exactly one nonexecuted resolved `y=0`
  counterfactual for each retry-exhausted lineage;
- `D- / realized_collision|realized_oob`: the one selected and executed `y=1`
  action whose immediate live transition realizes collision/OOB.

Each row must serialize context, H10 action, original flow base, gamma, round,
scenario, lineage, step, attempt, K/B indices, beta/ESS/sigma, prediction audit,
and full verifier result. Never train on all B queries, relabel an exact
negative, or reintroduce P1/P2/Ncausal/D0 roles. A realized-failure row must
retain `verification.valid=true`; its separate realized label determines its
negative training role.

Optimize

\[
\mathcal L(\theta)=
\operatorname{mean}_{D^+}\mathcal L_{\rm CFM}
-\alpha\operatorname{mean}_{D^-}\mathcal L_{\rm CFM}.
\]

`alpha=0` is mandatory as a control. Keep positive/negative objective mass
explicit; report sample counts, unique exposure, duplicates, optimizer steps,
gradient norms/cosines, clipping, CFM losses, and model drift. A rare D- must
not silently receive arbitrary mass through oversampling.

The authoritative trainable set is:

```text
policy.trunk.inp
policy.trunk.blocks[0]
policy.trunk.blocks[1]
policy.head
```

Freeze `grid_conv`, `grid_projection`, low encoder, GRU/history encoder, and all
other condition encoders. Assert their state SHA before and after every round.
The architecture has two residual blocks; this surface is the entire flow
trunk plus head.

The no-update diagnostic initializes RBF support from exactly 50 balanced
pretrained embeddings. For the first end-to-end qualification, preserve that
preflight. If you introduce round-to-round GP support, declare its cap, window,
gamma/lineage/time quotas, and frozen/current representation before running;
do not silently inherit an old Hp10 or P1/P2 buffer rule.

## 5. Required implementation gates

Before any long run, add tests and prove:

1. strict checkpoint/config/SHA admission and cached r0 equivalence;
2. identical clipped scene dynamics and exact-verifier behavior;
3. K=64, B=32, without-replacement acquisition and ESS target .1;
4. retry std sequence and no state advance during retry;
5. selector ordering: progress, clearance, sigma, index;
6. only exact positives execute; those without immediate realized failure enter
   D+;
7. exactly one nonexecuted exact negative enters D- on exhaustion;
8. an immediate live collision/OOB moves the one executed `y=1` row to D-
   without altering its verifier label;
9. stored context/action/base/provenance round-trip exactly;
10. positive-minus-alpha-negative loss signs and mass;
11. complete flow trunk/head train while all condition encoders remain bitwise
    frozen;
12. acquisition gathering and independent raw evaluation cannot be confused;
13. one exact trace/video follows the committed event log without resampling.

Run a short no-update diagnostic first and compare it against the historical
max-margin trace. Report per gamma:

- B16 and B32 exact-positive counts and zero-positive frequency;
- retry attempts and terminal NVP;
- selected H10/one-step progress;
- selected CV clearance and exact certificate margin;
- selector disagreement with max-step-margin and each selector's progress,
  clearance, and time contribution;
- SR/CR/timeout of the acquisition controller, clearly labeled training only.

## 6. Expansion and evaluation funnel

After the gates pass:

1. Run a one-round qualification with `alpha=0` plus a small declared nonzero-
   alpha arm. Save r0 and r1.
2. Evaluate both using a fixed, disjoint raw M10 bank at temperature 1.
3. Inspect whether the updated policy reproduces selected positive actions at
   their stored contexts and whether D- likelihood decreases.
4. Only then run a bounded multi-round arm search. Recipes must be fixed before
   seeing their confirmation results.
5. Shortlist on raw M10, evaluate shortlisted checkpoints on fresh M50, and
   evaluate the single locked winner on untouched M100.

The paper plot must use independently regenerated raw trajectories, not
acquisition/controller rollouts. Track:

- collision rate;
- window Validity;
- successful minimum clearance;
- successful time to goal;
- SR and timeout liveness guards;
- per-gamma values and uncertainty intervals.

Temperature-one is mandatory. Optional per-gamma temperatures require a
separate calibration bank, a frozen mapping, and a new confirmation bank.

Baseline raw M50:

| Distribution | SR | CR | Timeout | Validity | Clearance | Time |
|---|---:|---:|---:|---:|---:|---:|
| ID | .9571 | .0429 | 0 | .7906 | .3010 m | 5.617 s |
| OOD | .5600 | .4371 | .0029 | .4769 | .1086 m | 7.147 s |

The target is substantially lower OOD CR with higher Validity and successful
clearance while preserving liveness and meaningful gamma trends. These are
targets, not gates that may be manufactured by posthoc checkpoint, episode, or
temperature selection.

## 7. Visualization contract

Use the two historical clips only as layout/style templates:

- `assets/templates/acquisition_before_template.mp4`;
- `assets/templates/raw_after_template.mp4`.

For the new diagnostic, show K=64 lightly, B=32 distinctly, rejected candidates
red, exact positives blue, selected positive emphasized, exact verifier
polytope/level sets green, actual executed states black, goal visible, and
attempt/std/frame indices. Select several close-interaction contexts where
roughly 4-5 candidates are exact positive if the declared deterministic screen
finds them. Always include the complete screen ledger; do not hand-pick a case
without reporting the selection rule.

Write compact committed outputs under `assets/diagnostics/predictive_execution/`
only after validating them. Large trace tensors stay in the Helios output root.

## 8. Do not change

- canonical checkpoint, pretraining data, ID/OOD scene definitions, or fixed
  evaluation banks;
- policy observation or condition encoders;
- raw pedestrian state as verifier/selector-only information;
- exact GREEN verifier, 16 outer faces, full-H10 semantics, or clipped dynamics;
- K64/B32/ESS .1/retry schedule/selector ordering in the control protocol;
- raw temperature-one evaluator;
- another agent's checkout, branch, output root, or artifacts.

Do not add Kazuki, privileged MPC, artificial positive labels, hidden fallback,
recovery starts, max-margin execution, native-cost scalarization, or posthoc
confirmation tuning to the authoritative control.

## 9. Historical protocol — retained only for comparison

The original HP100 port used conditional repair after a raw proposal failed,
initially `K=16/B=4`, max-one-step-margin selection, head-only learning, and
multiple P1/P2/Dminus/Ncausal/D0 replay roles. Later diagnostics used K64/B16,
weighted native cost, additional trunk scopes, and cumulative micro-rounds.
Those modules and the two template videos are historical evidence. They do not
define the SFM2 update.

The old Hp10/B1 material is under the files prefixed `LEGACY_HP10_`.

## 10. Required delivery

Return:

- frozen origin/main SHA, your branch SHA, and clean-worktree proof;
- command/config and physical GPU UUID provenance;
- checkpoint, dataset manifest, scene, verifier, evaluator, and source hashes;
- focused/full test results and all invariant checks;
- no-update predictive diagnostic plus trace-faithful MP4/snapshots/statistics;
- per-round D+/D-, beta/ESS, retries/NVP, replay accounting, loss/gradient/drift;
- raw r0 and every saved checkpoint's M10 results;
- shortlisted disjoint M50 and one untouched final M100 result;
- four-metric/per-gamma plots from raw fixed-bank evaluation;
- a SHA-256 manifest of every delivered artifact;
- honest failure/timeout/deviation report.

Stop fail-closed on a checkpoint, source, scene, verifier, serialization,
freeze, or raw-evaluation mismatch. Do not launch the long sweep until the
one-round qualification is complete and audited.

## Copy-paste prompt for Claude

> Clone `https://github.com/DHLeexpress/safe_flow_expansion_SFM2.git` from
> `origin/main` into a new private local checkout, record the frozen SHA, create
> your own `agent/claude-sfm2-predictive-YYYYMMDD` branch, and read
> `README.md` and `CLAUDE_HP100_EXPANSION_HANDOFF.md` completely. Run
> `python scripts/show_claude_hp100_handoff.py`,
> `python scripts/verify_package.py`, and the focused tests before editing.
> Preserve the canonical HP100 checkpoint/data, ID/OOD scenes, clipped
> dynamics, policy inputs, fixed raw evaluator, and exact full-H10 GREEN
> verifier. Starting additively from
> `source_snapshot/overnight_run_07_12_sfm/sfm_hp100_predictive_execution.py`,
> implement end-to-end always-on Safe Flow Expansion: at every replan generate
> K=64, uncertainty-acquire B=32 without replacement at normalized ESS .1,
> exact-verify B, and execute an exact positive selected lexicographically by
> H10 goal progress, CV-predicted clearance, selected sigma, then candidate
> index. If B is all negative, hold state and retry with Gaussian flow-base std
> `1.0,1.1,...`; after the declared limit execute none, archive one resolved
> exact-negative counterfactual, and terminate NVP. If an executed exact-
> positive first action immediately realizes collision/OOB in live SFM, retain
> its verifier `y=1` but archive that one executed row as a realized D- and
> terminate. Train only on the remaining selected executed positives and the
> two declared negative sources with
> `L = mean(L_CFM(D+)) - alpha*mean(L_CFM(D-))`; include alpha=0. Train
> `trunk.inp + both residual blocks + head`, freeze every condition encoder,
> and prove all archive, loss-sign/mass, freeze, provenance, and raw-evaluation
> invariants. First deliver a no-update diagnostic and trace-faithful comparison
> video/statistics, then a one-round qualification with fixed raw M10. Only if
> it passes, run a bounded fixed-recipe search followed by fresh M50 shortlist
> evaluation and one untouched M100 confirmation. Use Helios GPUs only through
> your own processes, never kill or modify foreign jobs, and write all generated
> artifacts under a new
> `/data3/research1/claude_sfm2_predictive_<sha7>/` root. Commit/push only your
> private branch; report hashes, tests, GPU provenance, D+/D-/ESS/retry/replay
> accounting, raw per-round CR/Validity/clearance/time/SR/timeout, gamma trends,
> videos, manifests, deviations, and failures. Do not claim success unless the
> independent fixed-bank results establish it.
