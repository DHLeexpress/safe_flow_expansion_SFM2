#!/usr/bin/env bash
# Re-cherry-pick under the re-targeting crowd: collect PRE/Champion cases for
# the champion-all-success episodes, run Kazuki on them, render and assemble.
set -uo pipefail
cd "$HOME/projects/safe_flow_expansion_SFM2-claude-cfc09ad/source_snapshot/overnight_run_07_12_sfm"
PY="$HOME/miniforge3/envs/cfm_mppi/bin/python"
OUT=/data3/research1/claude_sfm2_predictive_cfc09ad
K=$OUT/kazuki_compare
V=$K/v3
FUN=$OUT/funnel/shortlist_m50_champ
EPS=${EPS:-920020,920027}
R0=$HOME/projects/safe_flow_expansion_SFM2-claude-cfc09ad/checkpoints/hp100_pretrained_r0_258999ae.pt
R0_SHA=258999ae8ccee8aec5aab92a6f751221d3c15583ac26e0a7ec8311f13316ec44
CH=$OUT/champions/XF_FULL_E4/checkpoint_r1.pt
CH_SHA=3d782897ac73455823d2d67382e7410d4f23ba469884548cd25c7dfe5a7ebebe
export PYTHONPATH=$PWD
mkdir -p $V/traces $V/panels
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }

stamp "collect PRE (GPU 0) + Champion (GPU 3) for $EPS"
( CUDA_VISIBLE_DEVICES=0 $PY collect_compare_ext.py \
    --checkpoint "$R0" --expected-sha "$R0_SHA" \
    --evaluation $FUN/r0_double_density_velocity_ood.json \
    --ep0 920000 --episodes "$EPS" --label pre_ext --output $V/traces \
    --device cuda:0 > $V/collect_pre.log 2>&1 || stamp "PRE collect FAILED" ) &
P1=$!
( CUDA_VISIBLE_DEVICES=3 $PY collect_compare_ext.py \
    --checkpoint "$CH" --expected-sha "$CH_SHA" \
    --evaluation $FUN/XF_FULL_E4_double_density_velocity_ood.json \
    --ep0 920000 --episodes "$EPS" --label champ_ext --output $V/traces \
    --device cuda:0 > $V/collect_champ.log 2>&1 || stamp "CHAMP collect FAILED" ) &
P2=$!
wait $P1 $P2
echo COLLECT_DONE

stamp "kazuki rollouts for $EPS (GPU 0)"
CUDA_VISIBLE_DEVICES=0 $PY run_kazuki_ext.py \
  --checkpoint "$R0" --episodes "$EPS" --gammas 0.1,0.5,1.0 \
  --safe-coef 0.7 --goal-coef 0.5 \
  --output $V/kazuki --device cuda:0 > $V/kazuki.log 2>&1 \
  || stamp "KAZUKI FAILED"
cat $V/kazuki/KAZUKI_SUMMARY.json 2>/dev/null
echo KAZUKI_DONE

stamp "render + assemble each episode"
DIMS=$($PY make_frame.py --panel-w 748 --panel-h 748 --left 170 --top 120 \
        --bottom 300 --output $V/frame.png 2>>$V/render.log)
set -- $DIMS
FW=$1; FH=$2
for EP in ${EPS//,/ }; do
  mkdir -p $V/panels_$EP
  $PY render_compare2.py \
    --pre-cases $V/traces/pre_ext_cases.pt \
    --champion-cases $V/traces/champ_ext_cases.pt \
    --kazuki-dir $V/kazuki --episode "$EP" --output $V/panels_$EP \
    >> $V/render.log 2>&1 || { stamp "RENDER $EP FAILED"; continue; }
  bash $K/assemble_grid2.sh $V/panels_$EP $V/compare_3x3_ep${EP}_v3.mp4 \
    $V/frame.png 170 120 "$FW" "$FH" >> $V/assemble.log 2>&1 \
    || stamp "ASSEMBLE $EP FAILED"
done
stamp "done"
echo VIDEO3_DONE
