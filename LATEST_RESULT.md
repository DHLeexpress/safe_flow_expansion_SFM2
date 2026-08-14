# Latest canonical result: HP100 pretrained r0

Date: 2026-08-02

## Verdict

The newly promoted HP100 model is a strong matched-ID raw policy and a useful,
non-saturated OOD starting point. It was trained from scratch on the corrected
SafeMPPI demonstration contract and selected without reading any OOD result.
Several HP100 expansion variants were subsequently tested, but none produced
an authenticated raw OOD improvement over this r0 baseline. The current work
therefore isolates the execution-rule mechanism before another gradient run.

| distribution | SR | CR | timeout | Validity | successful clearance | successful time |
|---|---:|---:|---:|---:|---:|---:|
| matched ID, M50/gamma | 95.71% | 4.29% | 0% | 79.06% | 0.301 m | 5.62 s |
| double-density/double-speed OOD, M50/gamma | 56.00% | 43.71% | 0.29% | 47.69% | 0.109 m | 7.15 s |

Both rows use the same raw contract: temperature 1, NFE 8, one sampled H10
window per context, first-action execution, and no verifier selection,
guidance, MPPI refinement, fallback, or privileged controller.

## Promoted model

- File: `checkpoints/hp100_pretrained_r0_258999ae.pt`
- SHA-256: `258999ae8ccee8aec5aab92a6f751221d3c15583ac26e0a7ec8311f13316ec44`
- Selected epoch: 119
- Architecture: `v3-sfm-hp100-residual`
- Input: 10x32x100 Hp history, GRU-16 robot-control history, low state
- Dynamics: componentwise action and velocity caps at 2 m/s2 and 2 m/s
- Geometry: current-position tangent, `predict_gain=0`, 16 nominal faces
- Promotion: ID validation plus fixed raw temperature-one ID banks only

The promoted container adds authenticated metadata, so its file hash is not
expected to equal the intermediate `ckpt_119.pt` container hash. The promoted
state dictionary and configuration are the selected epoch-119 model.

## Dataset correction

The new collection fills exactly 500 successful SafeMPPI lineages for every
gamma, for 3,500 total. It stores all context provenance but admits a CFM
target only when at least one of 2,048 expert proposals was accepted and the
accepted-set weighted H10 plan independently passes the same frozen nominal
polytope recheck. This leaves 201,075 eligible targets from 204,297 contexts.

The 32 angular observation rays and the 16 nominal-polytope faces are separate
objects. No predictive-retreat face is used, and the 10x32x100 grid is rebuilt
from raw state/pedestrian geometry rather than upsampled from Hp10.

## Current execution-rule qualification

The default-off diagnostic in `sfm_hp100_predictive_execution.py` replaces the
failed raw-first/max-margin repair controller with always-on K64/B32
uncertainty acquisition. Exact-positive candidates retain the full
gamma-conditioned GREEN gate; execution then maximizes H10 goal progress, with
constant-velocity predicted clearance and uncertainty only as deterministic
tie-breaks. No checkpoint is updated in this qualification.

Only after its trace, videos, and storage semantics pass is the planned replay
`L_positive - alpha L_negative`, using the full flow trunk and head while
freezing every condition encoder. The detailed fail-closed protocol is in
`CLAUDE_HP100_EXPANSION_HANDOFF.md`.

Success must be measured on independent raw evaluation, not the acquisition
controller: reduce OOD CR and increase Validity and successful clearance while
retaining liveness and preserving ID behavior. A per-gamma temperature study,
if used, requires a separate calibration bank followed by an untouched
confirmation bank; temperature-one raw results remain mandatory.

## Evidence

- `provenance/hp100_pretrain_20260802/pretraining_report.json`
- `provenance/hp100_pretrain_20260802/dataset_manifest.json`
- `provenance/hp100_pretrain_20260802/id_m50.json`
- `provenance/hp100_pretrain_20260802/ood_m50.json`
- `assets/hp100_20260802/branch_viz/hp100_id_ood_branch.mp4`
- `assets/hp100_20260802/expert_mechanism/hp100_safemppi_mechanism_g0p2_ep109.mp4`
- `assets/hp100_20260802/data_provenance/hp100_expert_g0p2_ep109.mp4`

The previous Hp10 result is preserved in
`LEGACY_HP10_RESULT_20260722.md`; it is not the current baseline.
