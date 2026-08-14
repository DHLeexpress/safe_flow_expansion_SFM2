#!/bin/bash
# Re-render overnight arms as legible 2x3 tree grids (6 trees/arm, one per 1k iter). User 2026-07-06.
cd /home/dohyun/projects/cfm_mppi/overnight_run_2026-07-02 || exit 1
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib:${LD_LIBRARY_PATH:-}
V=results/hp_arch/res2w256_ft_v2.pt
ckpts() { echo "results/hp_overnight/$1/ckpt_1000.pt results/hp_overnight/$1/ckpt_2000.pt results/hp_overnight/$1/ckpt_3000.pt results/hp_overnight/$1/ckpt_4000.pt results/hp_overnight/$1/ckpt_5000.pt"; }
L="v2 it1000 it2000 it3000 it4000 it5000"

render() {  # $1 arm $2 temp $3 beta $4 gpu
  CUDA_VISIBLE_DEVICES=$4 python hp_tree_viz.py --ckpts $V $(ckpts $1) --labels $L \
    --gamma 0.5 --temp $2 --beta $3 --ell 0.5 --ncols 3 --tag ${1}_grid --outdir figures/dr_test_overnight \
    > figures/dr_test_overnight/render_${1}.log 2>&1
  echo "[$1] $(grep -c reached figures/dr_test_overnight/render_${1}.log >/dev/null && tail -1 figures/dr_test_overnight/render_${1}.log || echo done)"
}

render ov_mine 1.5 0.1 3 &
render ov_s08  1.5 0.05 0 &
wait
render ov_aggr 2.0 2.0 3 &
render ov_conj 1.5 0.05 0 &      # diverged — may render degenerate/empty; captured either way
wait
echo "GRIDS_DONE"
ls -1 figures/dr_test_overnight/tree_ov_*_grid.png
