#!/bin/bash
# User correction 2026-07-06 02:0x: ov_aggr = temp 2.0 (not 1.5) + beta 2.0 + lr 1e-4.
# The overnight pipeline is mid-Phase-A and cannot be edited while bash executes it; this watcher
# intercepts the stale ov_aggr right after Phase D launches it, wipes it, relaunches corrected.
set -u
cd /home/dohyun/projects/cfm_mppi/overnight_run_2026-07-02 || exit 1
while ! pgrep -f "hp-ov-agg[r]" > /dev/null; do
  pgrep -f "overnight_pipelin[e]" > /dev/null || { echo "pipeline died before Phase D"; exit 1; }
  sleep 20
done
sleep 5
pkill -9 -f "hp-ov-agg[r]"
sleep 3
rm -rf results/hp_overnight/ov_aggr results/hp_overnight/ov_aggr.log
CUDA_VISIBLE_DEVICES=0 setsid nohup python grid_hp_expt.py --iters 5000 --measure-every 500 --n-measure 50 \
  --temp 2.0 --ell 0.5 --enc-lr-mult 0 --arch-ckpt results/hp_arch/res2w256_ft_v2.pt --wandb-mode disabled \
  --demo-frac 0.25 --lwf-eta 0 --alpha 0.1 --beta 2.0 --lr 1e-4 \
  --outdir results/hp_overnight/ov_aggr --name hp-ov-aggr2 > results/hp_overnight/ov_aggr.log 2>&1 &
echo "ov_aggr CORRECTED relaunch pid $! (temp 2.0, beta 2.0, lr 1e-4)"
sleep 20
grep -E "override (temp|beta|lr)" results/hp_overnight/ov_aggr.log
