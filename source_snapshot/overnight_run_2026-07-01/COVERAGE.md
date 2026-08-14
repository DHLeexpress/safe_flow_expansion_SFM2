# What is "coverage" for the windowed generative policy?

The policy's design space is the control window `U_local ∈ R^{H_pred×2}` (H_pred=10 → 20-D per window) deployed
receding-horizon, so a *behavior* is a whole closed-loop path. "Coverage" must be measured in a **low-dimensional
behavior / outcome space** (the rolled-out path), not the raw control space — grid coverage in `R^{20}`-per-window
(let alone the full path) is intractable and meaningless (the valid set is a thin manifold; many windows → the same
motion). Everything below is on the **rolled-out paths**, normalized by the **verifier-reachable-safe set Ω\***
(a broad proposal gated by the SAME `validity` module — you can only cover what is certifiable-safe-and-reaching).

## Metrics (`coverage.py`, all swappable)
1. **spatial_coverage (headline)** — fraction of Ω\*'s free-space cells that the FM's *verifier-valid* paths occupy.
   Robust, bounded `[0,1]`, and the number that must rise during expansion.
2. **mode_coverage** — fraction of Ω\*'s behavior **modes** the FM reproduces. Modes are scene-specific homotopy
   classes:
   - **gap**: `center` / `upper` / `lower` (which part of the corridor the thread passes).
   - **slalom**: `around_up` / `around_down` / `weave` (sweep above / below the pair vs weave between them).
   This is the direct "did expansion discover the new mode?" measurement (e.g. SafeMPPI gives `around_*`; expansion
   should add `weave`).
3. **vendi** — diversity (effective number of distinct behaviors) of a low-dim lateral-offset descriptor.
4. **validity** — fraction of the FM's raw sampled paths that are verifier-valid (efficiency, not safety; every
   *kept* path is exactly safe).

## Why this is the right altitude
- **Safety** is per-path and exact: `validity.py` = `collision_free ∧ reaches_goal ∧ SOCP-certified` (per γ). Only
  certified windows ever train the policy, so "coverage ↑" never trades away safety.
- **γ is single-sourced**: it enters the policy context (in `low_dim`) *and* the verifier ruler, so coverage is
  reported per γ and the γ-graded behavior (gap: center↔upper/lower) is visible.
- **Extensible**: `validity.CHECKS` and `coverage.mode_of` are lists/functions — add *goal-optimality*,
  *performant* (time/energy), or *comfort* checks later without touching the loop.

## Ω\* (the denominator)
`coverage.build_omega_star` runs a broad "surrounding" proposal (PD-to-waypoint with random lateral + weave),
gates it through the verifier at `γ_max`, and records the reachable-safe **cells / modes / descriptors**. This is
also the exploration support the finite-β Eq.9 objective deviates into (see README) — the same broad proposal seeds
the expansion buffer `D_0`.

## Success criterion (measured, per scene)
- **gap**: pretrained FM threads narrowly (few modes / low spatial cov) → expanded FM covers `center+upper+lower`
  at every γ, spatial coverage ↑, all certified.
- **slalom**: pretrained/SafeMPPI = `around_*` only → expanded FM adds `weave` (mode_coverage ↑ toward 1), spatial
  coverage of the between-obstacle region ↑, all certified.
