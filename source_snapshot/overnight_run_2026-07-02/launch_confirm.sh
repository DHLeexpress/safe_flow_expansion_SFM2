#!/bin/bash
# Phase-S confirm stage (helios GPU3 x2): 2k confirms of the R2 top-2. nyx runs the R3 interpolation screens.
cd /home/dohyun/projects/cfm_mppi/overnight_run_2026-07-02 || exit 1
BASE="--iters 2000 --ell 0.5 --measure-every 100 --n-measure 50 --enc-lr-mult 0 --demo-frac 0.5 --lwf-eta 0.1 --arch-ckpt results/hp_arch/res2w256_ft.pt --wandb-mode disabled"

CUDA_VISIBLE_DEVICES=3 setsid nohup python grid_hp_expt.py $BASE --temp 1.5 \
  --outdir results/hp_screen/confirm_combo --name hp-confirm-combo \
  > results/hp_screen/confirm_combo.log 2>&1 &
echo "confirm_combo pid $!"

CUDA_VISIBLE_DEVICES=3 setsid nohup python grid_hp_expt.py $BASE --temp 1.3 \
  --outdir results/hp_screen/confirm_delta05 --name hp-confirm-delta05 \
  > results/hp_screen/confirm_delta05.log 2>&1 &
echo "confirm_delta05 pid $!"
