# Retrospective NVP support audit

This read-only audit was run after the sweep. It did not modify any source,
checkpoint, query trace, GP state, replay data, or result marker. At every NVP
context for rounds 1–3 of `margin_alpha0p001_inner016`, it applied the same
frozen full-H verifier and nominal-\(H_P\) gate to the unqueried \(K-B=12\)
candidates already present in the saved trace.

| round | NVP contexts | acquisition miss: admissible candidate existed in unqueried 12 | no admissible candidate in all \(K=16\) | nominal-gate failure | solver error |
|---:|---:|---:|---:|---:|---:|
| 1 | 56 | 14 | 42 | 0 | 0 |
| 2 | 56 | 17 | 39 | 0 | 0 |
| 3 | 56 | 10 | 46 | 0 | 0 |

Thus checking all 16 candidates would have rescued only 18–30% of these NVP
contexts. The dominant observed category was exhaustion of the saved learned
\(K=16\) pool, not the nominal gate. This cannot by itself distinguish scarcity
of learned proposals from genuine state/H10 infeasibility; adaptive \(K=64\)
is the next falsification test.

Audit identity:

- source: `safeMPPI@e5ab47ba4971aae6c1df710c6d6864577f3728f7`;
- verifier `sfm_metrics2.py`: SHA-256
  `beb49fde691fdc3e86759cbd00ca45186ac350d5b088dae14061f29d5d290eca`;
- selector `sfm_b1_cost.py`: SHA-256
  `9020d73f79bcd0d4c45a61aab62f32237292faa68f76827ac25497ca19b32493`;
- saved r1 trace: SHA-256
  `31b34d3388be31598f34c22e2d4e11a30bc1ba232a8885ef623085f95c7c6918`;
- saved r2 trace: SHA-256
  `82a42fab12570bbeae8bea9790a6fc1dd91c86c81dfd1e980120b60e18c1a27c`;
- saved r3 trace: SHA-256
  `b53247f7a8356f11d3d7bfd4c67900812977074fd106acaf2f2a4b1a1a1d6d1a`.

Because NVP is a censored finite-sample event here, this audit does not justify
labeling the immediately preceding certified action as a negative training
target. The source traces remain in the authenticated external run root; this
repository preserves their hashes and compact per-round aggregates, not the
large trace tensors.
