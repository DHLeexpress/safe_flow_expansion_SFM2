#!/bin/bash
# LOCAL half of the two-machine split (2026-07-05): aggressive search on GPU 0 & 3.
# Remote half (fine-tuning brackets) = HP_RUNBOOK.md matrix rows marked REMOTE.
cd /home/dohyun/projects/cfm_mppi/overnight_run_2026-07-02 || exit 1
mkdir -p results/hp_dist
BASE="--temp 1.5 --ell 0.5 --enc-lr-mult 0.5 --measure-every 100 --arch-ckpt results/hp_arch/res2w256_ft.pt --wandb-mode disabled"

CUDA_VISIBLE_DEVICES=0 setsid nohup python grid_hp_expt.py --iters 2000 $BASE \
  --demo-frac 0.25 --outdir results/hp_dist/dfrac0.25 --name hp-dfrac0.25 \
  > results/hp_dist/dfrac0.25.log 2>&1 &
echo "dfrac0.25 pid $!"

CUDA_VISIBLE_DEVICES=3 setsid nohup python grid_hp_expt.py --iters 2000 $BASE \
  --lwf-eta 0.1 --outdir results/hp_dist/lwf0.1 --name hp-lwf0.1 \
  > results/hp_dist/lwf0.1.log 2>&1 &
echo "lwf0.1 pid $!"

CUDA_VISIBLE_DEVICES=0 setsid nohup python grid_hp_expt.py --iters 1000 $BASE \
  --lr 1e-5 --outdir results/hp_sweep6/lr1e-5 --name hp6-lr1e-5 \
  > results/hp_sweep6/lr1e-5.log 2>&1 &
echo "lr1e-5re pid $!"
