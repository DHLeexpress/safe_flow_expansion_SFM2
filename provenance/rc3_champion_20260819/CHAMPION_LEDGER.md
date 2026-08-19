# RC3_12500 — certified champion ledger (2026-08-19)

## Checkpoint

- Path (Helios): `/data3/research1/claude_sfm2_predictive_cfc09ad/champions/RC3_12500/snapshot_step12500.pt`
- SHA-256: `f0253de394cea2bbdfe31c4fe5a3fc23bac9ae8ebbc7f109369b094c45b5a748`
- Strict `{state_dict, config}` payload, loadable by the frozen evaluator.

## Recipe (declared, reproducible)

Raw-path one-shot from canonical r0 on the tagged 3-rule mixed corpus
(308,361 D+ / 1,043 D-): existing lam4/r.45 104k + lam4/r.60 100k +
lam8/r.45 104k extended collections (raw observations + shadow, fp16 shards).
`sfm_hp100_raw_train.py`: optimizer_scope=all_open (1,978,068 params;
noise_templates pinned), context-path raw, alpha=0.02 hinge margin 2.0,
positive_mass=per_gamma_balanced, lr=1e-5, batch 64, exposure_passes 4,
train_mode eval, update seed 2, grad clip 1.0, drift gate 0.25,
lru-shards 200 (shard-residency fix), snapshot-every 2500.
Champion = the step-12,500 snapshot (~2/3 of the 19,272-step schedule; the
mid-schedule optimum replicated across three independent dose curves).

## Evidence chain

- M20 screen (fixed CRN bank, OOD ep0 900000): CR .329 / Val .710 /
  clr .119 / time 7.72 / SR .657 / TO .014 — campaign-best legal checkpoint.
- Fresh M50 (untouched, OOD ep0 920000 / ID 930000):
  OOD CR **.480** vs r0 .551 (**-7.1pp, -13% relative**), Val .691 (+5.0pp),
  clearance .102 (+1.6pp), SR .514 (+6.6pp), time 8.22 (+0.74s), TO .006;
  ID CR .040 / SR .954 / Val .773 (intact). Beats both prior champions on
  the same bank (XF_FULL_E4 .509, W2_E8 .506).
- Per-gamma OOD CR vs r0: 0.1 .50->.34, 0.7 .62->.48, 0.5 .56->.48 (the
  chronic weak band resolved), 0.2 -6pp, 0.3 -4pp, 0.4/1.0 flat.
- Seed replication (update seed 3), M20 at steps 10000/12500/15000:
  CR .357/.364/.350 — inside the .33-.36 recipe band; SR .63-.64, TO clean.
- Training audit: accepted, drift trunk_head 4.5% / grid_projection 8.1% /
  encoders 2.3%; encoders trained via the raw path (all_open), noise
  templates bitwise frozen.

## Key findings the champion rests on

1. MPC-cost acquisition rule (lam4/rho1.1/r_eff.45/sigma.1) turns the
   controller from 41% to 72-93% lineage success with zero executed
   collisions, and its data teaches lower CR than progress-argmax data.
2. Mixed multi-rule data resolves the per-gamma weak band that single-rule
   data cannot (gamma 0.4-0.5).
3. Full encoder opening + high dose is required to digest 300k-scale data;
   alpha-hinge (0.02) matters most at reduced capacity.
4. Train CFM loss is flat while real CR swings — validation must be
   snapshot raw-M20, never loss.
5. The historical multi-hour raw-path trainings were shard-deserialization
   bound; with shard residency the true compute is minutes, enabling this
   search density.

## Status

Fresh-M50-certified challenger. Final claim requires the declared untouched
M100 confirmation (ep0 940000/950000) after contract lock — pending the
user's go. Until then, per the handoff, this is a strong fixed-bank result,
not a completed protocol claim.
