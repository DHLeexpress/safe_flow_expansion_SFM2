# Night-4 offensive ledger — BoN-16 self-distillation champion (2026-08-20/21)

## Headline

**New fresh-M50 champion: N4A2_s12500.**
OOD CR **.391** (r0 .551 → **-16.0pp, -29% relative**; RC3_12500 .480 →
-8.9pp), Validity .724, successful clearance .126, SR .589, timeout .020,
time 8.61 (+0.39s vs RC3 — declared TtG trade).  ID bank improved, not just
intact: CR .020 / SR .980 / Val .817 / clearance .321 (RC3: .040/.954/.773).
Per-gamma OOD CR: 0.1 .24, 0.2 .34, 0.3 .42, 0.4 .38, 0.5 .48, 0.7 .52,
1.0 .36 — every gamma at or below .52, no collapsed band, all gammas at or
better than r0.  The lineage's final checkpoint independently lands at .397,
confirming the region.  Achieved by TRAINING ONLY — the N-sample deploy
switch stays off and unofficial.

## Checkpoint

- Path (Helios): `/data3/research1/claude_sfm2_predictive_cfc09ad/champions/N4A2_12500/snapshot_step12500.pt`
- SHA-256: `bb1c3e1544ccc4c9da8c40caeb3762047fb25f78bb3a327fe5b64500bf2e76f8`
- Strict `{state_dict, config}` payload, loadable by the frozen evaluator.

## Recipe (declared, reproducible)

Two stages:

1. **Teacher data — BoN-16 self-distillation corpus (339,026 D+).**
   `sfm_hp100_bon_distill_collect.py` (commit 1b94d58/37eef97): the prior
   champion RC3_12500 rolled closed-loop with best-of-16 MPC-select
   (tuned lam4/rho1.1/r_eff.45/sigma.1; j=0 = canonical CRN latent) on fresh
   OOD-profile scenes ep0 800000+/860000+ (disjoint from every eval bank),
   43 blocks, 6,020 episodes.  Teacher episode outcomes: SR .811,
   CR .152.  Every executed step of every SUCCESS episode is certified
   post-hoc by the exact full-H10 GREEN verifier; only valid steps become
   positives (87.4% pass), with raw-obs shards captured for encoder
   training.  Collision/timeout episodes contribute nothing.
2. **Student — champion-continue on distillation data only.**
   `sfm_hp100_raw_train.py` from RC3_12500: archives = 43 BoN blocks +
   negs_only (the 1,499 shared D-), optimizer_scope=all_open, context-path
   raw, alpha .02 hinge margin 2.0, positive_mass per_gamma_balanced,
   lr 5e-6, batch 64, exposure 4, train_mode eval, seed 2, grad clip 1.0,
   raw-audit-atol 10 (r0-relative negative manifests), snapshot-every 1250.
   Champion = step-12,500 snapshot of the 21,190-step schedule (~60% —
   the mid-schedule optimum, now replicated a 4th time).

## Evidence chain

- M20 screen (fixed CRN bank, ep0 900000): CR .329 / **Val .739
  (campaign-high)** / clr .1437 / SR .643 / TO .029; per-gamma CR max .40,
  gamma 0.4 at .15.
- Fresh M50 (ep0 920000/930000): the headline numbers above.  M20->M50
  shrinkage +6.2pp — far less winner's curse than RC3's +15.1pp on the same
  funnel.
- Sibling A-arm cross-check: N4A1 (r0 + 308k acquisition corpus + the same
  BoN corpus, champion recipe) reached fresh-M50 .451 — also beating RC3 —
  showing the BoN corpus, not the continue trick alone, carries the signal.
- Mode-sweep M50 control: MD_HARD_OPEN_s02500 (M20 .314) collapsed to .489
  on fresh M50 — the standing winner's-curse discipline caught it; data
  selection alone does not beat the champion.

## The night's three-prong verdict

| Prong | Arms | Verdict |
| --- | --- | --- |
| A. BoN-16 self-distillation | N4A1 (r0+full+BoN), N4A2 (RC3-continue on BoN) | **Both beat RC3 on fresh M50 (.451 / .391). A2 is the new champion.** |
| B. Hard-mode champion polish | N4B1/B2 (hard), N4B3 (avoid) RC3-continues | M20 .350-.386 — nothing beyond the champion band; confirms data-selection saturation. |
| C. DPO-v2 tuned volley | N4C1 (beta .3/anchor .25), N4C2 (beta 1/anchor .1/lr 2e-5) | M20 .457 / .521 (C2 timeout guard-fail). With the NFT pilot (B01 guard-fail, B10/CX0 = r0) and the displacement diagnostic (regime False), the contrastive-objective track is closed on evidence: the mechanism is clean but does not transfer CR on this task. |

## Interpretation

The blind-spot probe said the residual failures were a sampling/imitation
gap, not coverage; the fix that worked is exactly the one that attacks that
gap: on-policy states under the improved controller, selection-sharpened
targets (93.5% of steps chose a non-canonical proposal; 7% of replans
rescued a CV-predicted contact), and verifier-certified positives.  The
2.49-nat BoN-16 teacher tilt identified in the hazard analysis is now
partially in the weights: Validity rose to .72-.74 everywhere BoN data was
used, clearance rose ~+.02-.04, and OOD CR moved -8.9pp in one round.

## Incidents (recorded, both fixed in-flight)

- First GPU-0 collect died on an over-tight re-encode audit (1e-4);
  batched-vs-single kernel accumulation measures up to ~1.3e-3, so the
  audit atol moved to the campaign-standard 5e-3 with per-block max
  deviation recorded (37eef97).
- Two launcher one-liners hit `&&`/`&` precedence, silently dropping
  variables in backgrounded units; relaunched from committed script files
  (the standing lesson now applied to every launch).

## Status & open items

- N4A2_s12500 is a fresh-M50-certified champion.  The declared untouched
  M100 confirmation (ep0 940000/950000, contract lock) remains
  USER-GATED and has not been touched.
- Round 2 of the distillation loop (teacher = N4A2_s12500) is collecting
  as of this writing: arm N4A3, screen `screen_m20_N4c`.
- The N-switch (best-of-N at deploy) remains an unofficial final weapon;
  its calibration on the new champion is a cheap future measurement.
