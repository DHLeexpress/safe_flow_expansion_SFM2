#!/bin/bash
# PHASE-DR deployment (user 2026-07-06): frontier expansions from the DR-SPLICED model, encoder frozen.
# safeETA (δ.25 η1.0) GPU3 · safeDELTA (δ.75 η.1) GPU0 · temp 1.3 · n=50 · 2000 iters.
cd /home/dohyun/projects/cfm_mppi/overnight_run_2026-07-02 || exit 1
mkdir -p results/hp_dr
BASE="--iters 2000 --temp 1.3 --ell 0.5 --measure-every 100 --n-measure 50 --enc-lr-mult 0 --arch-ckpt results/hp_arch/res2w256_dr.pt --wandb-mode disabled"

CUDA_VISIBLE_DEVICES=3 setsid nohup python grid_hp_expt.py $BASE \
  --demo-frac 0.25 --lwf-eta 1.0 \
  --outdir results/hp_dr/dr_safeETA --name hp-dr-safeETA \
  > results/hp_dr/dr_safeETA.log 2>&1 &
echo "dr_safeETA (GPU3) pid $!"

CUDA_VISIBLE_DEVICES=0 setsid nohup python grid_hp_expt.py $BASE \
  --demo-frac 0.75 --lwf-eta 0.1 \
  --outdir results/hp_dr/dr_safeDELTA --name hp-dr-safeDELTA \
  > results/hp_dr/dr_safeDELTA.log 2>&1 &
echo "dr_safeDELTA (GPU0) pid $!"
