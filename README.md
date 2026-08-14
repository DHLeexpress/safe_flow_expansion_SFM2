# Safe Flow Expansion for moving-crowd SFM navigation

This is the clean handoff repository for the HP100 moving-pedestrian study.
The canonical pretrained policy, scene, dynamics, and exact verifier are
frozen. The active research question is whether **always-on uncertainty
acquisition with prediction-aware progress execution** can improve the raw
generative policy without teaching the long, conservative detours produced by
the former max-margin selector.

> **Scientific status.** The acquisition mechanism below is implemented as a
> no-update diagnostic. No long expansion run has yet established an SR, CR,
> Validity, clearance, or time-to-goal improvement. The protocol is the next
> experiment, not a claimed result.

## Start here

| Item | Link | Purpose |
|---|---|---|
| Claude handoff | [CLAUDE_HP100_EXPANSION_HANDOFF.md](CLAUDE_HP100_EXPANSION_HANDOFF.md) | Immutable contracts, implementation task, and copy-paste prompt |
| Always-on acquisition | [sfm_hp100_predictive_execution.py](source_snapshot/overnight_run_07_12_sfm/sfm_hp100_predictive_execution.py) | K=64/B=32 acquisition, exact verification, prediction-aware selector, and trace archive |
| Mechanism tests | [test_sfm_hp100_predictive_execution.py](source_snapshot/overnight_run_07_12_sfm/analysis/test_sfm_hp100_predictive_execution.py) | Selector ordering, retry budget, and archive semantics |
| Exact GREEN verifier | [sfm_metrics2.py](source_snapshot/overnight_run_07_12_sfm/sfm_metrics2.py) | Candidate-specific full-H10 moving-pedestrian paper-SOCP verifier |
| HP100 policy | [grid_policy_sfm_hp100.py](source_snapshot/overnight_run_07_12_sfm/grid_policy_sfm_hp100.py) | Frozen condition encoders and trainable flow trunk/head |
| Canonical checkpoint | [hp100_pretrained_r0_258999ae.pt](checkpoints/hp100_pretrained_r0_258999ae.pt) | Promoted epoch-119 HP100 model |
| Dataset pointer | [DATA_POINTER.json](DATA_POINTER.json) | Helios tensor location, hashes, split, and target semantics |
| Raw evaluator | [sfm_hp100_eval.py](source_snapshot/overnight_run_07_12_sfm/sfm_hp100_eval.py) | Fixed raw temperature-one evaluation |
| Four-metric report | [sfm_hp100_early_eval_report.py](scripts/sfm_hp100_early_eval_report.py) | CR, Validity, successful clearance, and successful time-to-goal curves |
| Before/repair template | [acquisition-before MP4](assets/templates/acquisition_before_template.mp4) | Old conditional-repair visual template; not the new mechanism |
| After/raw template | [raw-after MP4](assets/templates/raw_after_template.mp4) | Same-lineage raw-policy visual template |
| New diagnostic result | [mechanism report](assets/diagnostics/predictive_execution/MECHANISM_RESULT.md) | K64/B32 statistics, limitations, comparison video, rollout, and montage |
| Full acquisition audit | [2 episodes x 3 gamma MP4](assets/diagnostics/predictive_execution/predictive_acquisition_2episodes_g0p1_g0p5_g1p0.mp4) | Complete K64 -> B32 -> exact verification -> prediction-based execution trajectories |

The two template videos are presentation references. Their old controller and
sample semantics must not be attributed to the new protocol.

## Canonical baseline

The promoted checkpoint is authenticated by SHA-256
`258999ae8ccee8aec5aab92a6f751221d3c15583ac26e0a7ec8311f13316ec44`.
It was selected with trajectory-disjoint ID validation only; OOD never entered
checkpoint promotion.

Canonical raw evaluation means one unguided H10 flow sample per context,
temperature 1, NFE 8, first clipped action execution, and no GP, verifier
selector, Kazuki guidance, MPPI refinement, fallback, or privileged lookahead.

| Distribution | Pedestrians / speed | SR | CR | Timeout | Validity | Successful clearance | Successful time |
|---|---|---:|---:|---:|---:|---:|---:|
| Matched ID | 20 / 0.5-1.0 m/s | 95.71% | 4.29% | 0% | 79.06% | 0.301 m | 5.62 s |
| Double-shift OOD | 40 / 1.0-2.0 m/s | 56.00% | 43.71% | 0.29% | 47.69% | 0.109 m | 7.15 s |

The raw baseline artifacts are
[ID M50](provenance/hp100_pretrain_20260802/id_m50.json) and
[OOD M50](provenance/hp100_pretrain_20260802/ood_m50.json).

## Authoritative always-on acquisition

At **every** closed-loop replan, rather than only after a raw temperature-one
proposal fails:

1. Generate `K=64` learned H10 proposals, retaining each original Gaussian
   flow base.
2. Compute the frozen paired-noised penultimate representation

   \[
   z_j=\operatorname{normalize}\phi_{\theta_{\rm ref}}
   ((1-s)x_{0,j}+sU_j,s,c),\qquad s=0.9.
   \]

3. Use the RBF posterior and adaptive beta to acquire `B=32` without
   replacement at normalized `ESS/K=0.1`.
4. Resolve all B candidates using the exact full-H10 GREEN verifier at the
   current gamma.
5. Execute one exact-positive candidate using the lexicographic selector below.

For candidate \(j\), let

\[
\Delta_g^j=\lVert p_t-g\rVert-\lVert p^j_{t+H}-g\rVert
\]

be H10 goal progress. Using exact sensed pedestrian positions and velocities,
the independently audited constant-velocity clearance is

\[
c_{\rm CV}^j=
\min_{h,i}\left(
\lVert p^j_{t+h}-(q_t^i+h\Delta t\,v_t^i)\rVert-r_{\rm ped}
\right).
\]

Eligibility remains the gamma-dependent exact label \(y_j^\gamma=1\). Among
eligible B candidates, choose

\[
j^\star=\operatorname*{arg\,max}^{\rm lex}_{j:y_j^\gamma=1}
\left(\Delta_g^j,c_{\rm CV}^j,\sigma_j,-j\right).
\]

Thus uncertainty supplies informative queries, the exact verifier supplies
gamma-conditioned safety, H10 progress is the primary execution objective,
predicted clearance and uncertainty break ties, and candidate index gives a
deterministic final tie. There is no scalarized native-cost/max-margin weight.
Exact positivity already includes the constant-velocity collision check;
clearance is an audit and tie-break, not a second permissive gate.

If a B block contains no exact positive, hold the state fixed and retry with
Gaussian flow-base standard deviation

\[
1.0,1.1,1.2,\ldots,1.0+0.1(A-1),
\]

for the declared maximum number of attempts. This changes the flow base scale,
not the ODE sampling temperature. The authoritative diagnostic default is
32 attempts, \(a=0,\ldots,31\). On exhaustion, do not execute a negative:
archive exactly one resolved negative counterfactual from the final B and end
that lineage NVP.

### Full two-episode acquisition audit

The [2 x 3 acquisition video](assets/diagnostics/predictive_execution/predictive_acquisition_2episodes_g0p1_g0p5_g1p0.mp4)
runs paired OOD scenarios `40192750` and `755357831` from start to terminal at
`gamma in {0.1, 0.5, 1.0}`. Every panel shows the K=64 flow population, the
uncertainty-acquired B=32 subset, exact-negative endpoints, the
prediction-selected exact-positive H10 action and its candidate-specific GREEN
verifier, and the executed closed-loop path.

| gamma | Success | Collision | NVP | Timeout | Total |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 1/2 (50%) | 1/2 (50%) | 0/2 | 0/2 | 2 |
| 0.5 | 1/2 (50%) | 0/2 | 1/2 (50%) | 0/2 | 2 |
| 1.0 | 2/2 (100%) | 0/2 | 0/2 | 0/2 | 2 |
| **Pooled** | **4/6 (66.67%)** | **1/6 (16.67%)** | **1/6 (16.67%)** | **0/6** | **6** |

These are acquisition-controller lineage outcomes, not raw-policy evaluation
and not an expanded-checkpoint result. Six lineages are sufficient to audit the
mechanism and its failures, but not to estimate deployment performance; the
latter still requires the independent raw M10/M50/M100 protocol.

After an exact-positive first action is executed, advance the live SFM state.
If that immediate transition realizes collision or out-of-bounds despite its
pre-execution exact label, terminate and archive that executed row as a
realized negative. Preserve both facts: its verifier label remains \(y=1\),
while its realized outcome is collision/OOB.

## Expansion archive and update

The new archive has only two mutually truthful roles:

- **positive:** a selected and executed exact-positive H10 action whose
  immediate live-SFM transition does not realize collision/OOB;
- **negative (all-negative NVP):** one nonexecuted, resolved \(y=0\)
  counterfactual from the final all-negative B after retry exhaustion;
- **negative (realized failure):** the one selected and executed \(y=1\)
  action when its immediate live-SFM transition realizes collision/OOB.

Every record must retain context, action window, original flow base, gamma,
round, scenario and lineage identity, attempt/index, uncertainty, CV prediction
audit, and the complete exact-verifier result. Keep the verifier label and
realized outcome in separate fields: realized failure changes the training
role, not the historical verifier result. Do not archive all B rows as
training examples or revive P1/P2/Ncausal/D0 semantics.

The update objective is

\[
\mathcal L(\theta)=
\mathbb E_{D^+}\!\left[\mathcal L_{\rm CFM}\right]
-\alpha\,
\mathbb E_{D^-}\!\left[\mathcal L_{\rm CFM}\right].
\]

`alpha=0` is the required control. Any nonzero alpha, replay count, learning
rate, GP support rule, or round count is an explicitly reported experimental
choice. Sample exposure, duplicate exposure, optimizer steps, clipping, and
per-role loss must be audited.

The trainable surface is the complete flow trunk and output head:

```text
policy.trunk.inp
policy.trunk.blocks[0]
policy.trunk.blocks[1]
policy.head
```

The visual encoder, grid projection, low-state encoder, GRU/history encoder,
and every other condition encoder remain frozen. The model has two residual
blocks; “three blocks + head” means the trunk input layer plus both residual
blocks plus head, not three residual blocks.

## Immutable model, scene, and safety contracts

- Model: `v3-sfm-hp100-residual`, H10 output, componentwise action cap 2.0.
- Observation: `10x32x100` newest-to-oldest Hp history, 0.02 m radial cells,
  visual token 128, GRU-16 control history, low token 48.
- Robot: start `(0,0)`, goal `(6,6)`, `dt=0.1`, componentwise velocity cap 2.0.
- Gammas: `{0.1,0.2,0.3,0.4,0.5,0.7,1.0}`.
- OOD expansion scene: 40 pedestrians at 1.0-2.0 m/s.
- Nominal expert geometry: current-tangent, `predict_gain=0`, exactly 16 outer
  faces. The 32 observation rays are not polytope faces.
- Expansion truth: candidate-specific full-H10 GREEN verification with clipped
  dynamics, all-pedestrian CV collision checking, analytic H1-H10 faces, and
  exactly 16 artificial outer faces.

The flow policy does not receive raw pedestrian positions or velocities. They
are verifier/selector-only information refreshed from the SFM state after each
executed first action. This distinction must remain explicit in every claim.

## Dataset provenance

SafeMPPI collection retained exactly 500 successful ID lineages per gamma,
3,500 total. The authenticated dataset contains 204,297 contexts and 201,075
eligible weighted H10 targets with a globally trajectory-disjoint train/ID-val
split. Objective mass is balanced gamma -> successful lineage -> window.

Large tensors stay on Helios:

```text
/data3/research1/sfm_hp100_certified_weighted_500x7_2671a94
```

Use [DATA_POINTER.json](DATA_POINTER.json) and
[dataset_manifest.json](provenance/hp100_pretrain_20260802/dataset_manifest.json)
to authenticate them. Source provenance is:

- dataset collection: `2671a9447b7b914053dce5fe9be2a0aae6c67a8d`;
- pretraining: `e9164e5a6e70b86cecae4660e7732f8ecc6a93f7`;
- exact branch renderer: `b659526`;
- inherited integrated snapshot: `2473b01de65d5ea9b7383549c3fdf3ae10938fc4`.

## Evaluation and reporting

Acquisition trajectories are training data, not evaluation. Every saved
checkpoint must be evaluated with independent raw sampling. Temperature-one
results are mandatory. Per-gamma temperature calibration, if studied, must use
a separate calibration bank, be frozen, and then be measured once on a fresh
confirmation bank.

Track at least:

1. collision rate;
2. window Validity;
3. successful minimum clearance;
4. successful time to goal;
5. SR and timeout as liveness guards.

Use a staged funnel: fixed disjoint raw M10 for development, fresh M50 for
shortlisted checkpoints, and untouched M100 only for the final winner. The
primary OOD target is lower CR and higher Validity/clearance than r0 without
collapsing SR or time to goal. Report per-gamma trends and uncertainty; do not
select on the final confirmation bank.

## Reproducibility

```bash
python scripts/show_claude_hp100_handoff.py
python scripts/verify_package.py
pytest -q
```

The committed full-acquisition video is reproduced without resampling by:

```bash
cd source_snapshot/overnight_run_07_12_sfm
TRACE=/data3/research1/sfm2_predictive_execution_20260814_v2/predictive_trace.pt
python sfm_hp100_predictive_execution_viz.py grid \
  --trace "$TRACE" \
  --output ../../../assets/diagnostics/predictive_execution/predictive_acquisition_2episodes_g0p1_g0p5_g1p0.mp4 \
  --gammas 0.1 0.5 1.0 \
  --replicas 0 1 \
  --fps 5 \
  --frame-stride 1
```

The complete operational handoff is
[CLAUDE_HP100_EXPANSION_HANDOFF.md](CLAUDE_HP100_EXPANSION_HANDOFF.md).

## Historical protocol — not the current default

Earlier HP100/B1 studies used conditional repair only after a raw proposal
failed, `K=16/B=4` in the first port, max-one-step-margin or weighted native
cost execution, head-only updates, and multiple P1/P2/Dminus/Ncausal/D0 replay
roles. Those settings remain valuable negative controls and visualization
provenance, but they are **not** the authoritative SFM2 protocol.

The older Hp10 archive is separately labeled in
[LEGACY_HP10_B1_README.md](LEGACY_HP10_B1_README.md). The static-obstacle sister
repository is [DHLeexpress/safe_flow_expansion](https://github.com/DHLeexpress/safe_flow_expansion).
