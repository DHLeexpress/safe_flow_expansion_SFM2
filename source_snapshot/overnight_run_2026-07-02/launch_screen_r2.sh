#!/bin/bash
# Phase-S Round 2, helios half (GPU3 x2): relax-toward-exploration from the R1 validity-holders.
# nyx runs r2_combo (GPU1) + r2_etaT15 (GPU0).
cd /home/dohyun/projects/cfm_mppi/overnight_run_2026-07-02 || exit 1
BASE="--iters 500 --ell 0.5 --measure-every 100 --n-measure 50 --enc-lr-mult 0 --arch-ckpt results/hp_arch/res2w256_ft.pt --wandb-mode disabled"

CUDA_VISIBLE_DEVICES=3 setsid nohup python grid_hp_expt.py $BASE --temp 1.5 \
  --demo-frac 0.75 --lwf-eta 0.1 \
  --outdir results/hp_screen/r2_temp15 --name hp-r2-temp15 \
  > results/hp_screen/r2_temp15.log 2>&1 &
echo "r2_temp15 pid $!"

CUDA_VISIBLE_DEVICES=3 setsid nohup python grid_hp_expt.py $BASE --temp 1.3 \
  --demo-frac 0.5 --lwf-eta 0.1 \
  --outdir results/hp_screen/r2_delta05 --name hp-r2-delta05 \
  > results/hp_screen/r2_delta05.log 2>&1 &
echo "r2_delta05 pid $!"
