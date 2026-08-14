#!/bin/bash
# PHASE-DR milestone reporter (user 2026-07-06): $1 = iteration (500 | 1000).
# Waits for both deployment ckpts at that iter -> trends (helios) into figures/dr_test/
# -> trees on nyx (base + ckpts rows, temp 1.3) -> PNGs fetched back to figures/dr_test/.
set -u
M=$1
cd /home/dohyun/projects/cfm_mppi/overnight_run_2026-07-02 || exit 1
# NB: conda lib path ONLY for python (matplotlib CXXABI); it breaks ssh/scp (OpenSSL mismatch)
PYLIB=/home/dohyun/miniforge3/lib
NYX=dohyunlee@dhcp-101-145.caltech.edu
RD=results/hp_dr

while [ ! -f $RD/dr_safeETA/ckpt_$M.pt ] || [ ! -f $RD/dr_safeDELTA/ckpt_$M.pt ]; do
  pgrep -f "hp-dr-saf[e]" > /dev/null || { echo "RUNS_DEAD_BEFORE_$M"; exit 1; }
  sleep 90
done
echo "MILESTONE_${M}_REACHED $(date +%H:%M)"

LD_LIBRARY_PATH=$PYLIB python hp_trend_viz.py --log $RD/dr_safeETA.log --out dr_test/dr_safeETA_it${M}_trend.png \
  --title "dr_safeETA (d.25 e1.0 EF, spliced) @it$M" 2>/dev/null
LD_LIBRARY_PATH=$PYLIB python hp_trend_viz.py --log $RD/dr_safeDELTA.log --out dr_test/dr_safeDELTA_it${M}_trend.png \
  --title "dr_safeDELTA (d.75 e.1 EF, spliced) @it$M" 2>/dev/null

ssh -o BatchMode=yes $NYX "mkdir -p ~/projects/cfm_mppi/overnight_run_2026-07-02/$RD/dr_safeETA ~/projects/cfm_mppi/overnight_run_2026-07-02/$RD/dr_safeDELTA"
scp -o BatchMode=yes $RD/dr_safeETA/ckpt_*.pt  $NYX:~/projects/cfm_mppi/overnight_run_2026-07-02/$RD/dr_safeETA/
scp -o BatchMode=yes $RD/dr_safeDELTA/ckpt_*.pt $NYX:~/projects/cfm_mppi/overnight_run_2026-07-02/$RD/dr_safeDELTA/

if [ "$M" = "500" ]; then EC="results/hp_dr/dr_safeETA/ckpt_500.pt";  EL="it500";
  DC="results/hp_dr/dr_safeDELTA/ckpt_500.pt"; DL="it500";
else EC="results/hp_dr/dr_safeETA/ckpt_500.pt results/hp_dr/dr_safeETA/ckpt_1000.pt"; EL="it500 it1000";
  DC="results/hp_dr/dr_safeDELTA/ckpt_500.pt results/hp_dr/dr_safeDELTA/ckpt_1000.pt"; DL="it500 it1000"; fi

ssh -o BatchMode=yes $NYX "cd ~/projects/cfm_mppi/overnight_run_2026-07-02 && \
  CUDA_VISIBLE_DEVICES=1 python3 hp_tree_viz.py --ckpts results/hp_arch/res2w256_dr.pt $EC \
    --labels dr-base $EL --gamma 0.5 --temp 1.3 --ell 0.5 --beta 0.1 \
    --tag dr_safeETA_it$M --outdir figures/dr_test 2>&1 | tail -2 && \
  CUDA_VISIBLE_DEVICES=0 python3 hp_tree_viz.py --ckpts results/hp_arch/res2w256_dr.pt $DC \
    --labels dr-base $DL --gamma 0.5 --temp 1.3 --ell 0.5 --beta 0.1 \
    --tag dr_safeDELTA_it$M --outdir figures/dr_test 2>&1 | tail -2"

scp -o BatchMode=yes "$NYX:~/projects/cfm_mppi/overnight_run_2026-07-02/figures/dr_test/tree_dr_safe*_it$M.png" figures/dr_test/
echo "MILESTONE_${M}_DONE"
grep "^it00*$M:" $RD/dr_safeETA.log
grep "^it00*$M:" $RD/dr_safeDELTA.log
