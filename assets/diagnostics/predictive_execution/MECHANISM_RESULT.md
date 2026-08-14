# Always-on predictive execution diagnostic

This is a **no-update acquisition-controller diagnostic**, not a raw-policy
evaluation and not an expanded checkpoint. It ran the canonical HP100 r0 on
the double-density/double-speed OOD scene for two paired scenarios across
`gamma in {0.1,0.3,0.5,1.0}`. Every context used K=64, RBF-acquired B=32,
normalized ESS target 0.1, and the exact full-H10 GREEN verifier.

## Main mechanism result

Across 345 closed-loop contexts, comparing the final attempted B block:

| Statistic | First 16 acquired | Full B=32 |
|---|---:|---:|
| Mean exact positives | 12.52 | 25.00 |
| Median exact positives | 16 | 32 |
| Zero-positive contexts | 18/345 (5.22%) | 3/345 (0.87%) |

B=32 therefore rescued 15 of the 18 contexts missed by the first 16 acquired
candidates in this same trace. This is the valid within-trace B ablation; the
historical C3 B16 figure (1.68% zero-positive) came from different contexts and
must not be substituted for it.

Among contexts where both selectors had an exact-positive choice, the new
progress selector differed from max-step-margin 87.72% of the time. It changed
the means as follows:

| Selected-window statistic | Max-step-margin | Predictive progress | Difference |
|---|---:|---:|---:|
| H10 goal progress | 1.064 m | 1.137 m | **+0.073 m** |
| CV predicted clearance | 0.452 m | 0.441 m | -0.011 m |
| one-step Hp margin | 0.443 | 0.433 | -0.010 |

The mechanism thus recovers meaningful progress while keeping exact
gamma-conditioned eligibility. It does not maximize clearance or step margin.

## Gamma-resolved final-attempt audit

| gamma | contexts | B16 zero | B32 zero | progress gain | selector changed |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 108 | 6.48% | 0% | +0.065 m | 87.04% |
| 0.3 | 30 | 10.00% | 6.67% | +0.078 m | 82.14% |
| 0.5 | 78 | 5.13% | 1.28% | +0.084 m | 88.31% |
| 1.0 | 129 | 3.10% | 0% | +0.073 m | 89.15% |

Gamma remains active through both the policy condition and exact verifier; it
is not erased by the execution selector.

## Outcomes and limitation

The eight acquisition lineages ended in 4 success, 3 NVP, and 1 collision.
The archive contains 341 positives and 4 negatives: three nonexecuted final-B
NVP counterfactuals plus one executed verifier-positive action whose live SFM
transition collided after the pedestrian deviated from the constant-velocity
forecast. The exact verifier label and realized outcome are stored separately.

This one collision is direct evidence that CV/full-H exact positivity is not a
guarantee under the reactive live SFM pedestrian dynamics. It is also why the
realized-failure negative is part of the next replay contract. No performance
claim should be made until an updated checkpoint passes independent raw
M10/M50/M100 evaluation.

## Visual evidence

- [Successful predictive rollout](predictive_progress_success_rollout.mp4)
- [Five fixed-screen selector cases](predictive_vs_max_margin_cases.mp4)
- [Five-case montage](predictive_vs_max_margin_cases_montage.png)

The case screen was frozen before rendering: current nearest pedestrian
clearance at most 0.5 m, at least four exact positives in B32, selectors differ,
new H10 progress exceeds max-margin by at least 0.1 m, and predicted clearance
is nonnegative. It then preserves represented-gamma diversity and fills by
progress gain/proximity rank.

Helios output:
`/data3/research1/sfm2_predictive_execution_20260814_v2`.
