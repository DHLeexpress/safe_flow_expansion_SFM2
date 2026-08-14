#!/bin/bash
# Two 20k-iter runs (user 2026-07-06). Save every 1k, measure every 500 (n=50), grad-clip 10 (α guard).
# GPU0 = "yours" (Claude recipe, warm-start from ov_mine ckpt_5000). GPU2 = "mine" (user recipe, s* calibrated, from v2).
# NB GPU1 occupied by another user -> using free GPU2 instead.
cd /home/dohyun/projects/cfm_mppi/overnight_run_2026-07-02 || exit 1
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib:${LD_LIBRARY_PATH:-}
mkdir -p results/hp_20k
COMMON="--iters 20000 --measure-every 500 --n-measure 50 --ckpt-every 1000 --temp 1.5 --ell 0.5 --enc-lr-mult 0 --lr 1e-4 --grad-clip 10 --wandb-mode disabled"

CUDA_VISIBLE_DEVICES=0 setsid nohup python grid_hp_expt.py $COMMON \
  --demo-frac 0.4 --lwf-eta 0.05 --beta 0.1 --alpha 0.02 --s 0.9 \
  --arch-ckpt results/hp_overnight/ov_mine/ckpt_5000_arch.pt \
  --outdir results/hp_20k/yours --name hp-20k-yours \
  > results/hp_20k/yours.log 2>&1 &
echo "yours (GPU0, warm-start 5k -> +20k) pid $!"

CUDA_VISIBLE_DEVICES=3 setsid nohup python grid_hp_expt.py $COMMON \
  --demo-frac 0.25 --lwf-eta 0.05 --beta 0.1 --alpha 0.05 --s 0.99 \
  --arch-ckpt results/hp_arch/res2w256_ft_v2.pt \
  --outdir results/hp_20k/mine --name hp-20k-mine \
  > results/hp_20k/mine.log 2>&1 &
echo "mine (GPU3, v2 -> 20k, s*=0.99) pid $!"
