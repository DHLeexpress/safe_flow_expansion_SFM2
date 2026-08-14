# HP100 SFM code index

All source paths below are relative to `source_snapshot/`. The snapshot keeps
the original directory structure because the implementation imports through
`_paths.py`.

## Canonical HP100 pipeline

| file | role | blind spot / warning |
|---|---|---|
| `overnight_run_07_12_sfm/grid_policy_sfm_hp100.py` | HP100 conditional flow policy and strict checkpoint loader. | HP100 checkpoints are incompatible with the Hp10 class. |
| `overnight_run_07_12_sfm/sfm_hp100_features.py` | Fresh 32-angle x 100-radial-bin Hp raster from state and crowd geometry. | 32 observation rays are not 32 polytope faces; support remains K=16. |
| `overnight_run_07_12_sfm/sfm_hp100_history.py` | Leak-free newest-to-oldest ten-frame history. | Pre-episode slots repeat the first frame. |
| `overnight_run_07_12_sfm/sfm_hp100_dynamics.py` | Shared componentwise-capped robot integration. | Update ordering is part of the scientific contract. |
| `overnight_run_07_12_sfm/stage2_hp100_data.py` | Fill-to-500-per-gamma SafeMPPI collection with exhaustive attempt ledger and target eligibility. | Failed lineages do not consume quota; ineligible context rows are provenance only. |
| `overnight_run_07_12_sfm/stage3_hp100_pretrain.py` | Globally trajectory-disjoint balanced CFM pretraining and ID-only checkpoint promotion. | Requires the external authenticated tensor dataset. |
| `overnight_run_07_12_sfm/sfm_hp100_eval.py` | Canonical raw temperature-one ID/OOD evaluation. | Acquisition/controller rollouts are not raw evaluation. |
| `overnight_run_07_12_sfm/sfm_hp100_branch_viz.py` | Four-case ID/OOD branch collection and exact GREEN rendering. | Branch label and trajectory-average Validity are different statistics. |
| `overnight_run_07_12_sfm/sfm_hp100_data_viz.py` | Post-collection reconstruction audit and Hp100 provenance video. | Does not rerun or reselect expert data. |
| `overnight_run_07_12_sfm/sfm_hp100_expert_mechanism_viz.py` | Full 2,048-proposal SafeMPPI accept/reject/weighted-target mechanism. | Display subset is deterministic; planner still evaluates all 2,048. |
| `overnight_run_07_12_sfm/sfm_hp100_kazuki.py` | Kazuki goal/safety guidance adapted to HP100 and shared clipped dynamics. | Guidance is a comparator, not the raw learned policy. |
| `overnight_run_07_12_sfm/sfm_hp100_kazuki_eval.py` | HP100 Kazuki evaluation entry point. | A locked matching-bank HP100 Kazuki M50 result is still required. |
| `overnight_run_07_12_sfm/sfm_hp100_predictive_execution.py` | Default-off, no-update K64/B32 always-on acquisition and prediction-aware execution diagnostic. | This proves selector mechanics only; it is not an expanded checkpoint or raw evaluation. |
| `overnight_run_07_12_sfm/sfm_hp100_predictive_execution_viz.py` | Trace-only mechanism and closed-loop renderer for the new selector. | It must consume an authenticated trace; rendering cannot resample or relabel candidates. |
| `overnight_run_07_12_sfm/sfm_hp100_ball_adapter.py` | HP100 policy/task adapter and declared expansion optimizer scopes. | `trunk_and_head` means the trunk input projection, both residual blocks, and head; condition encoders stay frozen. |

## Shared exact-safety dependencies

| file | purpose |
|---|---|
| `overnight_run_07_12_sfm/sfm_protocol.py` | Scene/profile and immutable scientific constants. |
| `overnight_run_07_12_sfm/sfm_metrics2.py` | Exact full-H moving-pedestrian GREEN verifier and window Validity. |
| `overnight_run_07_12_sfm/sfm_scene.py` | Matched-ID and double-density/double-speed OOD generation. |
| `overnight_run_07_12_sfm/sfm_b1_eval.py` | Shared raw closed-loop integration semantics. |
| `overnight_run_2026-07-01/verifier_polytope.py` | Exact fitted-polytope calculations. |
| `cfm_mppi/safegpc_adapter/safemppi.py` | SafeMPPI proposal/acceptance implementation used for collection. |
| `overnight_run_today/src/flow_policy.py` | Base conditional flow-matching policy. |

## Current execution-rule experiment

The active mechanism is deliberately separated from the earlier conditional
repair loop. At every replan it samples K=64, performs RBF uncertainty
acquisition without replacement to B=32, exact-verifies B, and selects among
the exact positives by the weight-free lexicographic key

`(H10 goal progress, predicted clearance, selected uncertainty, -index)`.

The constant-velocity collision test is already part of exact GREEN
positivity. Predicted clearance is consequently a deterministic tie-break and
audit, not a hidden second safety threshold. If B has no exact positive, the
same state is retried at Gaussian flow-base scales 1.0, 1.1, ...; exhausting
the declared attempts emits one resolved negative and terminates the lineage.
The diagnostic does not train a model. Its future replay contract is one
executed exact-positive per successful context and one negative only at an
exhausted context, with `L_positive - alpha L_negative`.

The requested trainable scope is `trunk_and_head`: all flow-trunk layers and
the output head are trainable, while the visual encoder, low-state encoder,
and control-history GRU remain frozen.

## Historical expansion reference code

The existing `sfm_b1_*` modules preserve the authenticated Hp10 B1 mechanism:
RBF acquisition, D/D+ storage, max-margin/cost selection, replay, curves, and
sweep scheduling. They are algorithm references, not an HP100-ready launcher.
The next implementation must add an HP100 path without silently loading the new
checkpoint through `grid_policy_sfm.py`.

Earlier HP100 trials used head-only, last-block, or last-two-block scopes and a
raw-first conditional repair mechanism. They remain provenance, not the
current execution-rule contract. Every new run must recalibrate RBF length
scale from 50 balanced HP100 embeddings; the old Hp10 ell is not portable.

## Tests

The HP100 suite lives in
`source_snapshot/overnight_run_07_12_sfm/analysis/`:

- `test_sfm_hp100_core.py`
- `test_sfm_hp100_dynamics.py`
- `test_sfm_hp100_eval.py`
- `test_sfm_hp100_branch_viz.py`
- `test_sfm_hp100_data_viz.py`
- `test_sfm_hp100_expert_mechanism_viz.py`
- `test_sfm_hp100_kazuki.py`
- `test_sfm_hp100_predictive_execution.py`
- `test_sfm_hp100_ball_port.py`
- `test_stage2_hp100_data.py`
- `test_stage3_hp100_pretrain.py`

`tests/test_workbook_contract.py` validates bundled hashes, metrics, source
surface, and legacy preservation without rerunning the expensive collection or
training job.

## Historical code

The complete prior index is preserved in `LEGACY_HP10_CODE_INDEX.md`. In
particular, `scripts/sfm_compare.py`, `scripts/sfm_snapshot.py`, and
`LOCAL_SIX_ROW_COMPARISON.md` are Hp10-specific until explicitly ported. Merely
changing their checkpoint path to HP100 is invalid.
