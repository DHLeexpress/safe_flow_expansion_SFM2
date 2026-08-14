#!/bin/bash
# Autonomous GRU chain (user 2026-07-06): wait nyx GRU fine-tune -> fetch ckpt -> origin overlay ->
# safe-flow expansion with Claude's recipe, grid+GRU encoder FROZEN (enc_lr_mult 0), on helios GPU2.
set -u
cd /home/dohyun/projects/cfm_mppi/overnight_run_2026-07-02 || exit 1
# NB: conda lib ONLY for python (matplotlib CXXABI); exporting it globally breaks ssh/scp OpenSSL.
PYLIB=/home/dohyun/miniforge3/lib
NYX=dohyunlee@dhcp-101-145.caltech.edu
NYXD='~/projects/cfm_mppi/overnight_run_2026-07-02'

# 1) wait for fine-tune
while true; do
  d=$(ssh -o BatchMode=yes -o ConnectTimeout=15 $NYX "grep -c '\[gru_ft\] DONE' $NYXD/results/hp_arch/gru_ft.log 2>/dev/null; pgrep -cf 'hp_finetune_gr[u]'" 2>/dev/null)
  dn=$(echo "$d" | sed -n 1p); al=$(echo "$d" | sed -n 2p)
  [ "${dn:-0}" -ge 1 ] && { echo "GRU_FT_DONE"; break; }
  [ "${al:-0}" -eq 0 ] && { echo "GRU_FT_DIED"; ssh -o BatchMode=yes $NYX "tail -6 $NYXD/results/hp_arch/gru_ft.log"; exit 1; }
  sleep 120
done
ssh -o BatchMode=yes $NYX "grep -E 'RESULT|DONE' $NYXD/results/hp_arch/gru_ft.log | tail -3"

# 2) fetch the fine-tuned GRU checkpoint
scp -o BatchMode=yes $NYX:$NYXD/results/hp_arch/res2w256_gru_ft.pt results/hp_arch/ || { echo "SCP FAILED"; exit 1; }

# 3) origin overlay (iteration 0 = fine-tuned base)
LD_LIBRARY_PATH=$PYLIB python hp_origin_overlay.py --ckpt results/hp_arch/res2w256_gru_ft.pt --n 16 --tag gru_ft 2>&1 | tail -1

# 4) safe-flow expansion — Claude recipe, grid+GRU FROZEN, from the GRU base (5k, ckpt 1k, measure 500)
mkdir -p results/hp_gru_expand
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=$PYLIB setsid nohup python grid_hp_expt.py \
  --iters 5000 --measure-every 500 --n-measure 50 --ckpt-every 1000 \
  --temp 1.5 --ell 0.5 --enc-lr-mult 0 --lr 1e-4 --grad-clip 10 \
  --demo-frac 0.4 --lwf-eta 0.05 --beta 0.1 --alpha 0.02 --s 0.9 \
  --arch-ckpt results/hp_arch/res2w256_gru_ft.pt \
  --outdir results/hp_gru_expand/gru_mine --name hp-gru-mine \
  > results/hp_gru_expand/gru_mine.log 2>&1 &
echo "GRU expansion launched (GPU2) pid $!"
sleep 25
grep -E "override enc_lr_mult|override s=|EXPANSION|it00000" results/hp_gru_expand/gru_mine.log | head
echo "PIPELINE_LAUNCHED"
