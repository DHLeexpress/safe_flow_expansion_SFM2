# Corrected Pillar-3 verifier on eval80 UCY

## What was corrected

The local query trajectory is one 10-step double-integrator MPPI forward pass. The previous global-path interpretation was wrong. The plot now uses one fixed robot-centered verifier polytope per selected trajectory, rendered with the same normalized `H_grid` level-set convention as `di_grid.py`.

## Summary

- `dataset`: ucy
- `ego_path`: /mnt/data/eval80_ego_ucy.pt
- `obs_path`: /mnt/data/eval80_obs_ucy.pkl
- `n_cases`: 25
- `gamma`: 0.5
- `sensing`: 2.0
- `horizon`: 10
- `n_base`: 16
- `num_samples`: 256
- `verifier_success`: 22
- `verifier_failure`: 3
- `verifier_success_rate`: 0.88
- `verifier_mean_time_ms`: 30.808725772782385
- `verifier_median_time_ms`: 12.987274500119383
- `verifier_max_time_ms`: 193.98768599967298
- `verifier_mean_lp_time_ms`: 11.159278409195394
- `mppi_mean_time_ms`: 9.684780959978525
- `mean_accepted`: 85.8
- `mean_rejected`: 170.2
- `nominal_certified_count`: 17
- `failures`: [{'ep': 79, 't': 56, 'reason': 'LP_infeasible'}, {'ep': 223, 't': 12, 'reason': 'LP_infeasible'}, {'ep': 194, 't': 27, 'reason': 'LP_infeasible'}]

## Optimization problem

For a rollout `q_i`, `i=0..H`, with `q_0=c`, the verifier solves for one polytope `P={x: A x <= b}`. Write `m_k=b_k-a_k^T c`. The normalized level-set is `H_P(x)=min_k (b_k-a_k^T x)/m_k`. The certification condition is `H_P(q_i) >= alpha_i`, `alpha_i=(1-gamma)^i`, which becomes the linear constraints

```text
a_k^T(q_i-c) <= (1-alpha_i) m_k,  i=1..H, k=1..F.
```

Base faces are bounded by the sensing inner K-gon: `m_k <= sensing*cos(pi/K)`. Obstacle faces use tangent support bounds `m_k <= n^T(o-c)-r-kappa*tau*max(0,n^T(v_robot-v_obs))`. For each sensed obstacle, a unit support normal `n` is selected by a fast 1-D search over the separating cone to maximize slack against the rollout. With normals fixed, the LP is

```text
maximize    sum_k m_k
subject to  1e-5 <= m_k <= upper_k
            a_k^T(q_i-c) <= (1-alpha_i)m_k.
```

This keeps the same `A,b,c,margins` representation used by the existing renderer, but the obstacle support normals are trajectory-specific, so it is less conservative than always using the radial nominal face.

## 3-D extension

The same certificate extends from `x in R^2` to `x in R^3` by replacing the planar normal set with a 3-D template normal set, e.g. axis normals plus an icosahedral normal dictionary.  The polytope representation remains `P={x: A x <= b}`, `m_k=b_k-a_k^T c`, and the level-set certificate remains

```text
a_k^T(q_i-c) <= (1-alpha_i)m_k,   alpha_i=(1-gamma)^i.
```

For spherical obstacles `(o,r)`, each obstacle support face uses

```text
m_k <= n^T(o-c)-r-kappa*tau*max(0,n^T(v_robot-v_obs)).
```

The trajectory-specific normal selection becomes a 2-D search on the unit sphere over the separating cone `n^T(o-c)>r`, or a finite support search over an icosahedral refinement.  With normals fixed, the LP over margins is unchanged.  Visualization changes from `H_grid(X,Y)` contours to 3-D isosurfaces or slice planes of `H_P(x)`.
