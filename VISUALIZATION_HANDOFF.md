# Visualization handoff — SFM2 comparison videos and paper figures

Everything needed to reproduce, restyle, or re-target the presentation
visuals: the PRE / Safe-Flow-Expansion / CFM-MPPI comparison video, the
gamma x time figure, and the standalone legend.  Written for whoever picks
up the figures next; it assumes the reader knows the expansion study but has
never rendered one of these clips.

Companion documents: [CLAUDE_HP100_EXPANSION_HANDOFF.md](CLAUDE_HP100_EXPANSION_HANDOFF.md)
(the frozen research contract) and the result ledgers under `provenance/`.

---

## 0. Read this first: the one non-obvious fact

**The frozen crowd stops walking.** `HumanAgent.social_force_step`
(`cfm_mppi.utils`, reached through `sfm_scene.advance_humans`) begins with

```python
if np.linalg.norm(self.goal - self.state) < 0.1:
    self.control = np.zeros(2); return          # parked forever
```

Pedestrian goals are uniform in `[-2, 8]^2` and double-OOD speeds are
1.0-2.0 m/s, so most of the 40 pedestrians arrive within 30-60 steps and
never move again.  Measured on ep920036 (180 steps, fraction of pedestrians
that moved, in sixths of the episode):

| | 1st | 2nd | 3rd | 4th | 5th | 6th |
|---|---:|---:|---:|---:|---:|---:|
| frozen simulator | .903 | .568 | .253 | .101 | .033 | **.025** |
| re-targeting fork | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |

A long clip therefore ends with a robot walking through a parked crowd, and
**late-episode "success" is partly an artefact of that freeze**: replaying
ep920036 with a live crowd flips the champion from success at all three
gammas to collisions at gamma 0.5 and 1.0.  Any future claim that reads a
video as evidence must account for this.

`source_snapshot/overnight_run_07_12_sfm/ped_extend.py` is the fork:

* a **scoped context manager** rebinds `HumanAgent.social_force_step` and
  restores it on exit — the frozen module is never edited (same discipline
  as the acquisition-selector patch);
* on arrival it re-draws a goal with the constructor's own rules: uniform in
  `[-2, 8]^2`, **>= 2.0 m from the robot goal (6, 6)** (so nothing can park
  on or loiter at the goal) and >= 1.5 m of travel, using the agent's own
  RNG so a scenario id still fixes the crowd;
* `probe_crowd.py` reproduces the table above in ~30 s on CPU.

**Rollouts under this fork are video-only.**  They are not the fixed-bank
evaluation and must never be quoted as a metric; every artifact carries
`crowd_variant: retargeting_pedestrians` and
`not_a_fixed_bank_metric: true`.

---

## 1. What exists today

| Artifact | Where |
|---|---|
| Final 3x3 video (ep920027) | `kazuki_compare/v3/compare_3x3_ep920027_final.mp4` (Helios), local `compare_videos_v3/` |
| Alternate 3x3 (ep920020, strongest champion contrast) | same directories, `..._ep920020_v3.mp4` |
| Original-episode record (ep920036 under the live crowd) | `kazuki_compare/v2/`, local `compare_videos_v2/` |
| gamma x time figure (2-column, in-panel legend) | [assets/comparison_v3/champion_gamma_time_3x2.pdf](assets/comparison_v3/champion_gamma_time_3x2.pdf) |
| Standalone legend | [assets/comparison_v3/legend_figure.pdf](assets/comparison_v3/legend_figure.pdf) |
| Ledger (episode choice, calibration, deviations) | [assets/comparison_v3/COMPARE_LEDGER_V3.json](assets/comparison_v3/COMPARE_LEDGER_V3.json) |
| First-generation video (frozen crowd, in-panel labels) | local `compare_videos/`, Helios `kazuki_compare/panels/` |
| Acquisition-mechanism figures (gamma probe, mode clusters) | local `sample_viz/` |

Helios root for everything: `/data3/research1/claude_sfm2_predictive_cfc09ad/kazuki_compare/`.

The delivered clips visualize **XF_FULL_E4**, the champion at the time of
rendering (`3d782897ac73455823d2d67382e7410d4f23ba469884548cd25c7dfe5a7ebebe`).
The study has since moved on — see §6.

---

## 2. Pipeline

Three stages, all runnable from
`~/projects/safe_flow_expansion_SFM2-claude-cfc09ad/source_snapshot/overnight_run_07_12_sfm`
with `PYTHONPATH=$PWD` and the `cfm_mppi` conda env
(`~/miniforge3/envs/cfm_mppi/bin/python`).

### 2.1 Collect (GPU)

| Script | Role |
|---|---|
| `collect_compare.py` | frozen-crowd replay of a stored M50 bank; asserts the replay reproduces the stored row |
| `collect_compare_ext.py` | the same replay **inside the re-targeting fork**; validates each case against its own replayed row (the stored row is no longer an oracle) and writes a 50-episode story ledger |
| `kazuki_run.py` | CFM-MPPI rollouts (guidance `v + 0.5*grad_goal + 0.3*rho_H*grad_CBF`, 200 generated -> top-10 -> 200 perturbations -> MPPI, warm start s=0.8) |
| `run_kazuki_ext.py` | `kazuki_run` under the same fork (2-line wrapper) |

Both collectors reuse the CRN noise from that checkpoint's own M50 json, so
the crowd variant is the only difference versus the fixed-bank rollout.

```bash
CUDA_VISIBLE_DEVICES=0 python collect_compare_ext.py \
  --checkpoint <ckpt> --expected-sha <sha256> \
  --evaluation <funnel>/<label>_double_density_velocity_ood.json \
  --ep0 920000 --episodes 920027 --label champ_ext \
  --output <workdir>/traces --device cuda:0
```

### 2.2 Render (CPU, matplotlib)

| Script | Output |
|---|---|
| `render_compare2.py` | the nine panel MP4s (style revision 2, no in-panel text beyond ticks and the safety badge) |
| `make_frame.py` | the transparent outside frame: column (gamma) titles, row (method) titles, legend |
| `scripts/video/assemble_grid2.sh` | tpad-clone shorter panels, `xstack` 3x3, pad, overlay the frame, scale to 1900 px, plus a last-frame PNG |
| `figure_champion_grid.py` | the gamma x time PDF (one method, rows = gamma, columns = two snapshots) |
| `figure_legend.py` | the standalone legend PDF (shares `figure_champion_grid.legend_handles`) |

Because the titles live in an **overlay**, restyling names or the legend
only requires re-running `make_frame.py` + the assembler — no re-render.

### 2.3 Drivers

`scripts/video/video2_driver.sh` (single episode) and `video3_driver.sh`
(multi-episode re-pick) chain collect -> kazuki -> render -> assemble and
write stage markers (`COLLECT_DONE`, `KAZUKI_DONE`, `RENDER_DONE`,
`VIDEO*_DONE`) that a monitor can poll.  `ws_sweep.sh` sweeps the Kazuki
safety coefficient on one episode.  They expect the assembler at
`$K/assemble_grid2.sh` where `K=/data3/.../kazuki_compare`; copy
`scripts/video/*.sh` there before launching.

---

## 3. Style contract (as delivered)

Video panels (`render_compare2.py`):

| Element | Value |
|---|---|
| Executed trajectory | PRE `#CF2626`, Safe Flow Expansion `#2E9A30`, CFM-MPPI `#BB3FC4` (white halo) |
| Executed H10 action window ("trail") | multi-step safe `#0057FF`, unsafe `#6B0C0C`, alpha .90, black boundary via `path_effects.Stroke` |
| Multi-step safety badge | 22.5 pt (1.5x), pad .42, border in the state colour |
| Tick numbers | 13.5 pt (1.5x the style's 9.0) |
| Panel margins | left .085, right .995, top .995, bottom .075 (compact grid) |
| Verifier polytope | `STYLE.VERIFIER_GREEN`, audit-only in every row |
| Kazuki guidance arrows | goal cyan `#00B7C3`, safety orange `#FF7F0E` |

Paper figure (`figure_champion_grid.py`) — the user's one-off request:

| Element | Value |
|---|---|
| Verifier actions | positive pure blue `#0000FF`, negative pure red `#FF0000`, alpha .80, black boundary |
| Executed trajectory | green `#2E9A30` |
| Badge | removed entirely |
| gamma / t captions | 29.3 pt; axis labels `x [m]`, `y [m]` 22 pt |
| In-panel legend | top-left panel, 15.0 pt |
| Columns | `t = 4 [s]` and `t = 8 [s]` |

Fonts come from `STYLE.apply_computer_modern_style()`.  Note: the Computer
Modern face has **no em dash** — use `-` in legend labels or you get a
missing-glyph box.

---

## 4. Cherry-picking: what is and is not achievable

`collect_compare_ext.py` scores all 50 bank episodes, so picking is a query,
not a search.  Under the live crowd on the OOD bank (ep0 920000) only three
episodes keep the champion successful at all three gammas:

| Episode | Champion | PRE | CFM-MPPI (w_s .7) |
|---|---|---|---|
| 920027 **(delivered)** | success 12.1 / 7.1 / 7.7 s | fails at gamma 0.1 | success, success, collision |
| 920020 | success 10.6 / 5.0 / 5.4 s | fails at all three | collision at all three |
| 920011 | success | never fails | — |

Two constraints could **not** be met honestly and should not be re-attempted
without new evidence:

1. **"CFM-MPPI succeeds only at gamma 0.1."**  Kazuki's guidance is
   gamma-independent; its per-gamma outcome is driven by the CFM base
   samples.  Sweeping `w_safe` over {0.3, 0.5, 0.7, 1.0, 1.4} never produced
   that pattern — on 920020 it collides everywhere even at 1.4, and on
   920027 `w_safe=0.5` inverts the pattern (collides at 0.1/0.5, succeeds at
   1.0).  The delivered clip uses `w_safe=0.7, w_goal=0.5`, a declared
   deviation from the locked default `(0.3, 0.5)`.
2. **"Champion valid at every step (badge always True)."**  Raw temperature-1
   sampling produces occasional invalid windows; no episode in the bank is
   valid throughout.  Delivered validity is .84-.91.

The gamma-mode story *does* hold and is the figure's strongest asset: on
920027 the champion detours wide at gamma 0.1 and runs nearly straight at
gamma 1.0, and at `t = 8 s` gamma 0.1 is still en route while 0.5 and 1.0
have already arrived.

---

## 5. Integrity rules

* Never edit the frozen simulator, verifier, evaluator, or policy modules —
  extend by scoped rebinding (`ped_extend.retargeting_pedestrians`,
  `install_v2_selector`) or by a subclass, and restore on exit.
* Anything rendered under the crowd fork is presentation material only.
  Metrics come from the fixed banks through `sfm_hp100_eval.py`.
* Record every calibration that departs from a locked default in the ledger
  next to the artifact (see `COMPARE_LEDGER_V3.json`), including the ones
  that failed.
* Large media stays on Helios; only compact figures and ledgers belong in
  Git.

---

## 6. The obvious next task

The clips visualize **XF_FULL_E4**, but the study's champion is now
**N4A2_s12500** (`provenance/night4_20260820/NIGHT4_LEDGER.md`): fresh-M50
OOD CR **.391** versus XF_FULL_E4's .509, ID improved rather than intact,
and no collapsed gamma band.  Regenerating the comparison against the
current champion is a parameter change, not new code:

```bash
CH=/data3/research1/claude_sfm2_predictive_cfc09ad/champions/N4A2_12500/snapshot_step12500.pt
CH_SHA=bb1c3e1544ccc4c9da8c40caeb3762047fb25f78bb3a327fe5b64500bf2e76f8
# 1. that checkpoint needs its own M50 json for the CRN noise (funnel shortlist-m50)
# 2. EPS=<episodes> bash scripts/video/video3_driver.sh   (edit CH/CH_SHA at the top)
# 3. python figure_champion_grid.py --champion-cases <workdir>/traces/champ_ext_cases.pt ...
```

Expect the episode ranking to change: a stronger champion survives more
episodes, which should make the joint story (champion succeeds, baselines
fail) easier to find, not harder.

Open items, in the order I would do them:

1. re-render the 3x3 and the gamma x time figure against N4A2_s12500;
2. decide whether the paper wants the honest column captions
   (`t = 7.1 s` for a terminated row) or the uniform `t = 8 [s]` used now;
3. if a "valid throughout" clip is ever required, it needs a deliberate
   search over more banks — it does not exist in ep0 920000.
