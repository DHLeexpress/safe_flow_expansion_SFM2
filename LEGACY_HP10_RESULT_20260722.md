# SFM result and next falsifiable arm — 2026-07-22

## Verdict

The moving-crowd goal has **not** been achieved. The selected
`margin_alpha0p001_inner016@r3` checkpoint produced the following untouched
canonical temperature-one M100-per-gamma result on
`double_density_velocity_ood`:

| SR | CR | timeout | trajectory-level \(V_{\rm safe}\) | successful clearance | successful time |
|---:|---:|---:|---:|---:|---:|
| 66.57% | 32.57% | 0.86% | 0.00% | 0.116 m | 8.85 s |

The locked per-gamma temperature vector was worse (SR 65.86%, CR 33.29%). The
target CR below 5% was not reached, and the result did not beat the earlier
pretrained or Kazuki measurements. Those earlier measurements used a different
M100 bank, so the completed run also cannot make a strict paired improvement
claim until r0 and Kazuki are evaluated on its exact final bank.

## What the nine-arm sweep actually established

- Replay exposure showed the clearest association with transient M10 changes;
  alpha had no consistent benefit and no alpha fixed the mismatch. Most
  low-exposure arms selected r0 itself, and the apparent M50 winner did not
  reproduce on M100.
- Negative replay at \(\alpha\in\{10^{-3},10^{-2}\}\) did not convert local
  verifier positives into raw closed-loop safety.
- Across the nine arms, 10,078 of 10,080 gathering lineages ended in NVP after
  a median of 8–9 actions. Nevertheless each arm stored roughly 40k full-H
  positive windows. The replay target was therefore dominated by locally safe
  prefixes from trajectories that did not remain executable.
- An audit of the selected arm's NVP contexts found no nominal-\(H_P\)-gate or
  solver failures. At rounds 1–3, verifying the unselected 12 plans rescued only
  10–17 of 56 NVP contexts; 39–46 contexts had no admissible candidate in the
  saved \(K=16\) pool. The dominant observed category is finite-pool exhaustion,
  not the max-margin tie-break; the audit cannot distinguish proposal scarcity
  from genuine state/H10 infeasibility.
- The recorded late-round “uplift” compares sequentially conditioned selected
  sigma with pre-acquisition pool sigma, so its negative value is not a clean
  acquisition comparison. Future diagnostics must log base-sigma uplift and
  pending-conditioned sigma separately.

## Next arm

The next single arm is
`margin_alpha0p001_inner016_adaptiveK64`.

It keeps the selected learning recipe and max-margin execution fixed. At each
context it draws 64 **learned flow** proposals, samples their query order
sequentially without replacement under the same RBF Gibbs acquisition, and
verifies four at a time. It stops at the first batch containing
a full-H verifier-positive, nominal-\(H_P\)-admissible plan and executes the
maximum one-step margin plan. NVP occurs only after the declared 64-query budget
is exhausted. Every actually queried, successfully resolved result is stored;
solver errors are logged separately and unqueried plans are not stored.
There is no expert, template, fallback, recovery start, or hand-crafted plan.

This is preferable to immediately labeling NVP predecessors as negative: NVP
is censored finite-sample evidence, not proof that the preceding certified
action was dynamically nonviable. Adaptive \(K=64\) specifically tests whether
the observed K16 exhaustion was proposal scarcity or state/H10 infeasibility.

Run five rounds first because the previous winner peaked at r3. The evaluation
must be canonical temperature one only and must compare r0 and the selected
adaptive checkpoint on the identical M100 CRN bank. Required diagnostics are
initial-batch success, rescue-query ranges 5–16/17–32/33–64, exhausted-K NVP,
gate failures, verifier errors, realized queries per context, lineage length,
base-sigma uplift, pending-conditioned sigma, and the usual D/D+, beta, ESS,
raw SR/CR/validity/clearance/time.

The completed source, results, and checkpoint hashes are in
`provenance/e5ab47b_alpha_epoch_sweep/`. The next-arm launcher and tests live in
the source snapshot and must preserve the old fixed-B path bitwise.
