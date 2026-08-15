# MPC-rule expansion — M20 screening ledger (2026-08-15)

All rows: fixed disjoint raw CRN M20 bank, OOD `double_density_velocity_ood`,
ep0 900000, noise seed 20260814, 7 gammas x 20 = 140 rollouts per checkpoint,
temperature 1, NFE 8, unmodified `sfm_hp100_eval.py`. Training-side controller
statistics never appear here. Guards are the SEARCH_CONTRACT values
(SR 4/140, timeout 2/140, time +0.5 s, per-gamma CR/Validity collapse 0.10).

Acquisition rule for the M-prefixed arms: `predictive_mpc_v2`
(lam=4, rho=1.1, r_eff=0.45 m, sigma_len=0.1; tuned on the mid-gamma
NVP-heavy lineages in `mpc_rule/grid_pass1` + `grid_pass2`, validated on all
56 lineages: 52 success / 4 NVP / 0 collision vs 23/33/0 for the
progress-argmax rule on the archive-runner noise stream 23 success / 33 NVP).
Update recipe both arms: alpha=0, lr=1e-5, batch 64, E=1, train_mode eval,
grad clip 1.0, drift gate 0.25, frozen 50-row r0 RBF support, cumulative
r1..r10 with persistent Adam and exact resume states. Deep =
`trunk_and_head` (327,956 params); Shallow = `last_block_and_head`
(137,236 params; trunk.inp and blocks[0] pinned bitwise).

| ckpt | CR | Validity | succ. clearance | succ. time | SR | timeout | guards |
|---|---:|---:|---:|---:|---:|---:|---|
| r0 | .443 | .636 | .1022 | 7.19 | .557 | 0 | PASS |
| ME1_r1 | .393 | .661 | .0873 | 7.51 | .607 | 0 | PASS |
| ME1_r2 | .414 | .681 | .0935 | 7.76 | .586 | 0 | time |
| ME1_r3 | .471 | .678 | .1036 | 7.63 | .529 | 0 | CR@0.1,1.0 |
| ME1_r4 | .443 | .683 | .0947 | 7.75 | .557 | 0 | time, CR@0.5 |
| ME1_r5 | .386 | .718 | .0918 | 8.05 | .614 | 0 | time |
| ME1_r6 | .436 | .720 | .0909 | 8.28 | .564 | 0 | time, CR@0.2,1.0 |
| ME1_r7 | .471 | .734 | .0986 | 8.41 | .514 | .014 | SR, time, CR@0.2,0.5,1.0 |
| ME1_r8 | .471 | .726 | .1040 | 8.91 | .521 | .007 | SR, time, CR@0.2,0.5,1.0 |
| ME1_r9 | .457 | .734 | .1053 | 8.71 | .529 | .014 | time, CR@0.2,0.5,1.0 |
| ME1_r10 | .493 | .726 | .0932 | 9.33 | .479 | .029 | SR, TO, time, CR@0.2,0.5,1.0 |
| ME1L_r1 | .414 | .655 | .0952 | 7.36 | .586 | 0 | PASS |
| ME1L_r2 | .414 | .664 | .0935 | 7.54 | .586 | 0 | PASS |
| ME1L_r3 | .393 | .682 | .0887 | 7.92 | .607 | 0 | time |
| ME1L_r4 | .436 | .667 | .0947 | 7.69 | .564 | 0 | CR@0.2,1.0 |
| ME1L_r5 | .450 | .677 | .1002 | 7.90 | .550 | 0 | time, CR@0.2,1.0 |
| ME1L_r6 | .464 | .690 | .0926 | 8.04 | .536 | 0 | time, CR@0.2,1.0 |
| ME1L_r7 | .464 | .703 | .0944 | 8.54 | .536 | 0 | time, CR@0.5 |
| ME1L_r8 | .464 | .706 | .0942 | 8.40 | .529 | .007 | time |
| ME1L_r9 | .421 | .714 | .0783 | 8.50 | .571 | .007 | time, CR@1.0 |
| ME1L_r10 | .479 | .702 | .0868 | 8.38 | .521 | 0 | SR, time, CR@0.2,0.5,1.0 |

Historical old-rule arms (same bank; all guard-blocked): E1_r2 .429/.611,
E4_r2 .471/.590, E16_r2 .500/.570, E1L_r2 .471/.633, E4L_r2 .450/.612,
E16L_r2 .464/.606 — see the screen_m20_stage1/1b/1d markers.

## Reading

- The MPC-rule data is what unlocked guard-passing improvement: the same E1
  recipe that was guard-blocked on progress-argmax data (E1_r2) passes with
  CR -5.0pp / Validity +2.5pp / SR +5.0pp at ME1_r1.
- Validity climbs monotonically with rounds on both arms (deep .661->.734);
  CR bottoms mid-trajectory (deep r5 .386, shallow r3 .393); successful time
  rises monotonically and diverges late (deep +1.8 s by r10, timeouts appear
  from r7). Continued training on safety-selected data accumulates
  conservatism: the optimum is mid-trajectory, which is why every round's
  checkpoint and resume state were kept.
- Deep dominates shallow at the respective peaks.
- Pareto set for promotion: ME1_r1 (all guards pass) and ME1_r5
  (best CR .386 / Validity .718, fails only the +0.5 s time guard at +0.86 s).
- These are M20 screening numbers with ~.08-wide Wilson intervals; nothing
  here is a claimed result until fresh M50 and untouched M100 confirm it.

Raw artifacts (Helios): /data3/research1/claude_sfm2_predictive_cfc09ad/
{mpc_rule, mpc_expansion, funnel/screen_m20_*}.
