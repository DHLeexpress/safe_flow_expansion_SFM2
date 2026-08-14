#!/bin/bash
# LOCAL wave 2 (2026-07-05): combined mechanism arm + winner long-run.
cd /home/dohyun/projects/cfm_mppi/overnight_run_2026-07-02 || exit 1
BASE="--temp 1.5 --ell 0.5 --enc-lr-mult 0.5 --measure-every 100 --arch-ckpt results/hp_arch/res2w256_ft.pt --wandb-mode disabled"

CUDA_VISIBLE_DEVICES=0 setsid nohup python grid_hp_expt.py --iters 2000 $BASE \
  --demo-frac 0.25 --lwf-eta 0.1 --outdir results/hp_dist/dfrac0.25_lwf0.1 --name hp-combined \
  > results/hp_dist/dfrac0.25_lwf0.1.log 2>&1 &
echo "combined pid $!"

CUDA_VISIBLE_DEVICES=3 setsid nohup python grid_hp_expt.py --iters 5000 $BASE \
  --demo-frac 0.25 --outdir results/hp_dist/dfrac0.25_5k --name hp-dfrac0.25-5k \
  > results/hp_dist/dfrac0.25_5k.log 2>&1 &
echo "longrun pid $!"
