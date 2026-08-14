#!/bin/bash
# Phase-S Round 1, helios half (GPU3 x2): safeMAX + safeDELTA. nyx runs safeETA + safeNOEF.
cd /home/dohyun/projects/cfm_mppi/overnight_run_2026-07-02 || exit 1
mkdir -p results/hp_screen
BASE="--iters 500 --temp 1.3 --ell 0.5 --measure-every 100 --n-measure 50 --arch-ckpt results/hp_arch/res2w256_ft.pt --wandb-mode disabled"

CUDA_VISIBLE_DEVICES=3 setsid nohup python grid_hp_expt.py $BASE \
  --enc-lr-mult 0 --demo-frac 0.75 --lwf-eta 1.0 \
  --outdir results/hp_screen/safeMAX --name hp-safeMAX \
  > results/hp_screen/safeMAX.log 2>&1 &
echo "safeMAX pid $!"

CUDA_VISIBLE_DEVICES=3 setsid nohup python grid_hp_expt.py $BASE \
  --enc-lr-mult 0 --demo-frac 0.75 --lwf-eta 0.1 \
  --outdir results/hp_screen/safeDELTA --name hp-safeDELTA \
  > results/hp_screen/safeDELTA.log 2>&1 &
echo "safeDELTA pid $!"
