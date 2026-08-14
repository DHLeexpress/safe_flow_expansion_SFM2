# Repository lineage

## Current canonical milestone (2026-08-02)

The standalone SFM package now promotes the HP100 v3 pretrained baseline:
10x32x100 current-tangent Hp history, 16 nominal support faces, shared clipped
action/velocity dynamics, and no predictive retreat. Its tiered source lineage
is dataset collection `2671a944`, pretraining `e9164e5`, exact branch renderer
`b659526`, and integrated snapshot `ef35f1f`. The earlier Hp10/B1 lineage below
is retained as history, not as the current model contract.

## Answer to “when did the two repositories branch?”

They did not branch in Git. There is no common Git ancestor between
`DHLeexpress/safe_flow_expansion` and `DHLeexpress/safe_flow_expansion_SFM`.
The latter remote was empty before this handoff.

The **conceptual task fork** occurred in the shared `safeMPPI` repository at:

```text
e6fcebe278d076459df379c4fc0739e9cd18acff
2026-07-20 00:48:01 PDT
Implement SFM Hp10 B1 study
```

This was 4 h 38 min after the static standalone workbook's initial commit
`8ae3d99` (2026-07-19 20:09:39 PDT). The static and SFM lines were then updated
in parallel. Therefore neither repository should merge the other's Git history
as if one were a normal branch.

## Shared mechanism

Both repositories implement:

- conditional flow proposals over H=10 control windows;
- uncertainty-tilted finite-budget verifier acquisition;
- deterministic full-window labels and separate D/D+ bookkeeping;
- fail-closed execution from verified candidates;
- recent-window hierarchical replay;
- raw, untilted evaluation isolated from the gathering controller.

## Task-specific divergence

| component | static sister | SFM repository |
|---|---|---|
| dynamics / scene | double integrator, fixed circular obstacles | robot plus moving SFM pedestrians |
| observation | low7 + 32×32 nominal-polytope embedding | canonical HP100: 10×32×100 nominal signed-field history + low state + control GRU; legacy Hp10 used 10×16×12 |
| global modes | U/R routes around fixed obstacle | left/right/yield interaction modes across scenarios |
| verifier | fitted static trajectory polytope / SOCP-oriented checker | moving-face H10 certificate under constant-velocity pedestrians |
| uncertainty memory | static B1 RBF mechanism | RBF W2, cap512, per-gamma q75 retention |
| execution | static nominal-Hp max margin | moving-scene nominal-Hp max margin among full-H positives |
| OOD | enlarged/fused static obstacle | 40 pedestrians at 1–2 m/s vs 20 at .5–1 m/s |

## SFM commit milestones in `safeMPPI`

| commit | date (PDT) | change |
|---|---|---|
| `e6fcebe` | Jul 20 00:48 | Hp10+B1 implementation begins |
| `103476d` | Jul 20 02:15 | frozen selection and negative-replay semantics |
| `b2caf9a` | Jul 20 12:07 | explicit ID/OOD evaluation |
| `60be313` | Jul 20 23:05 | exact K16 verifier shared by expansion and visualization |
| `b27df76` | Jul 20 23:25 | matched-ID/double-shift deployment and visual provenance |
| `f0142ff` | Jul 21 02:13 | matched Kazuki cost; optimizer steps exposed |
| `d112ad2` | Jul 21 11:53 | full-H replay and gamma-balanced GP memory revision |
| `47800df` | Jul 21 12:28 | bounded runtime and `/data3/research1` artifact contract |
| `e5ab47b` | Jul 21 12:51 | eight scheduler slots for the current sweep |

At the 2026-07-22 remote audit, the sister repository's default `main` was at
`5c0b753`; its paper branches were `paper/b1-evolution-and-modes@e03be78` and
`paper/b1-margin50-trends@edc1c49`. The SFM standalone `master` was still at
`1820687` before this completed-result update. This temporal overlap means the
two projects were actively updated in parallel, not that either commit belongs
to the other repository's history. Static metrics and task-specific temperature
or adaptive-gamma studies are not evidence for the SFM scene and are not copied
here.
