#!/bin/bash
# OVERNIGHT AUTONOMOUS PIPELINE (user 2026-07-06, "trust me", ~6h budget). Phases:
# A: +2001 off-diagonal DR trajs (|y-x|>=0.5) appended into windows_g*.pt (backup first) + viz
# B: finetune res2w256_ft -> res2w256_ft_v2.pt on the merged 4002 trajs (GPU3)
# C: v2 it0 measure (n=50) on GPU3 · ship v2+data to nyx · v2 base tree on nyx GPU1
# D: 4 expansion arms x 5000 iters (helios GPU3: ov_conj+ov_s08 · GPU0: ov_aggr+ov_mine)
# E: 6-row trees for all arms on nyx (2 GPUs) + trends -> figures/dr_test_overnight/ · OVERNIGHT_DONE
set -u
cd /home/dohyun/projects/cfm_mppi/overnight_run_2026-07-02 || exit 1
PYLIB=/home/dohyun/miniforge3/lib
NYX=dohyunlee@dhcp-101-145.caltech.edu
NYXD='~/projects/cfm_mppi/overnight_run_2026-07-02'
FIGD=figures/dr_test_overnight
mkdir -p $FIGD results/hp_overnight
echo "=== OVERNIGHT START $(date) ==="

# ---------- PHASE A ----------
mkdir -p dataset/backup_2001traj
for g in 0.1 0.5 1.0; do cp -n dataset/windows_g$g.pt dataset/backup_2001traj/; done
echo "[A] backup done; generating 667 OD seeds/gamma"
LD_LIBRARY_PATH=$PYLIB python gen_dr_data.py --seeds 667 --offdiag 0.5 --out-prefix "" --append \
  > $FIGD/od_datagen.log 2>&1 || { echo "[A] DATAGEN FAILED"; exit 1; }
tail -3 $FIGD/od_datagen.log
LD_LIBRARY_PATH=$PYLIB python od_viz.py >> $FIGD/od_datagen.log 2>&1
echo "[A] done $(date)"

# ---------- PHASE B ----------
echo "[B] finetuning res2w256_ft_v2 on merged 4002 trajs (GPU3)"
CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=$PYLIB python hp_finetune_v2.py > $FIGD/ft_v2.log 2>&1 \
  || { echo "[B] FT FAILED"; tail -5 $FIGD/ft_v2.log; exit 1; }
tail -3 $FIGD/ft_v2.log
echo "[B] done $(date)"

# ---------- PHASE C ----------
echo "[C] v2 it0 measure (n=50, GPU3)"
CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=$PYLIB python grid_hp_expt.py --iters 1 --n-measure 50 \
  --measure-every 100 --temp 1.5 --ell 0.5 --arch-ckpt results/hp_arch/res2w256_ft_v2.pt \
  --outdir results/hp_overnight/it0_v2 --name it0-v2 --wandb-mode disabled 2>/dev/null \
  | grep "^it00000" | tee $FIGD/v2_it0.txt
scp -o BatchMode=yes results/hp_arch/res2w256_ft_v2.pt $NYX:$NYXD/results/hp_arch/
scp -o BatchMode=yes dataset/windows_g0.1.pt dataset/windows_g0.5.pt dataset/windows_g1.0.pt $NYX:$NYXD/dataset/
ssh -o BatchMode=yes $NYX "cd $NYXD && mkdir -p figures/dr_test_overnight && \
  CUDA_VISIBLE_DEVICES=1 python3 hp_tree_viz.py --ckpts results/hp_arch/res2w256_ft_v2.pt --labels v2-base \
  --gamma 0.5 --temp 1.5 --ell 0.5 --beta 0.1 --tag v2_base --outdir figures/dr_test_overnight 2>&1 | tail -2"
scp -o BatchMode=yes $NYX:$NYXD/figures/dr_test_overnight/tree_v2_base.png $FIGD/ || true
echo "[C] done $(date)"

# ---------- PHASE D ----------
B="--iters 5000 --measure-every 500 --n-measure 50 --temp 1.5 --ell 0.5 --enc-lr-mult 0 --arch-ckpt results/hp_arch/res2w256_ft_v2.pt --wandb-mode disabled"
echo "[D] launching 4 arms $(date)"
CUDA_VISIBLE_DEVICES=3 setsid nohup python grid_hp_expt.py $B --demo-frac 0.25 --lwf-eta 0.1 --beta 0.05 --alpha 0.05 \
  --outdir results/hp_overnight/ov_conj --name hp-ov-conj > results/hp_overnight/ov_conj.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 setsid nohup python grid_hp_expt.py $B --demo-frac 0.25 --lwf-eta 0.1 --beta 0.05 --alpha 0.05 --s 0.8 \
  --outdir results/hp_overnight/ov_s08 --name hp-ov-s08 > results/hp_overnight/ov_s08.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 setsid nohup python grid_hp_expt.py $B --demo-frac 0.25 --lwf-eta 0 --alpha 0.1 --beta 2.0 \
  --outdir results/hp_overnight/ov_aggr --name hp-ov-aggr > results/hp_overnight/ov_aggr.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 setsid nohup python grid_hp_expt.py $B --demo-frac 0.4 --lwf-eta 0.05 --beta 0.1 --alpha 0.02 --lr 1e-4 \
  --outdir results/hp_overnight/ov_mine --name hp-ov-mine > results/hp_overnight/ov_mine.log 2>&1 &
sleep 30; echo "[D] procs: $(pgrep -cf 'hp-ov-')"
while pgrep -f "hp-ov-[a-z]" > /dev/null; do sleep 300; done
echo "[D] all arms finished $(date)"
for a in ov_conj ov_s08 ov_aggr ov_mine; do echo "== $a"; tail -2 results/hp_overnight/$a.log; done

# ---------- PHASE E ----------
echo "[E] trees on nyx"
for a in ov_conj ov_s08 ov_aggr ov_mine; do
  ssh -o BatchMode=yes $NYX "mkdir -p $NYXD/results/hp_overnight/$a"
  scp -o BatchMode=yes results/hp_overnight/$a/ckpt_1000.pt results/hp_overnight/$a/ckpt_2000.pt \
    results/hp_overnight/$a/ckpt_3000.pt results/hp_overnight/$a/ckpt_4000.pt results/hp_overnight/$a/ckpt_5000.pt \
    $NYX:$NYXD/results/hp_overnight/$a/ || echo "[E] scp $a incomplete"
done
tree_cmd() {  # $1 arm $2 beta $3 gpu
  echo "cd $NYXD && CUDA_VISIBLE_DEVICES=$3 python3 hp_tree_viz.py --ckpts results/hp_arch/res2w256_ft_v2.pt results/hp_overnight/$1/ckpt_1000.pt results/hp_overnight/$1/ckpt_2000.pt results/hp_overnight/$1/ckpt_3000.pt results/hp_overnight/$1/ckpt_4000.pt results/hp_overnight/$1/ckpt_5000.pt --labels v2 it1000 it2000 it3000 it4000 it5000 --gamma 0.5 --temp 1.5 --ell 0.5 --beta $2 --tag $1_6row --outdir figures/dr_test_overnight 2>&1 | tail -1"
}
ssh -o BatchMode=yes $NYX "$(tree_cmd ov_conj 0.05 1); $(tree_cmd ov_s08 0.05 1)" &
ssh -o BatchMode=yes $NYX "$(tree_cmd ov_aggr 2.0 0); $(tree_cmd ov_mine 0.1 0)" &
wait
scp -o BatchMode=yes "$NYX:$NYXD/figures/dr_test_overnight/tree_ov_*_6row.png" $FIGD/ || echo "[E] tree fetch incomplete"
for a in ov_conj ov_s08 ov_aggr ov_mine; do
  LD_LIBRARY_PATH=$PYLIB python hp_trend_viz.py --log results/hp_overnight/$a.log \
    --out dr_test_overnight/${a}_trend.png --title "$a 5k (v2 base)" 2>/dev/null
done
touch OVERNIGHT_DONE
echo "=== OVERNIGHT DONE $(date) ==="
