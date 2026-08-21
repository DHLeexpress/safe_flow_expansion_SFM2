#!/usr/bin/env bash
# Kazuki safety-coefficient sweep on one episode (video story calibration).
set -uo pipefail
cd "$HOME/projects/safe_flow_expansion_SFM2-claude-cfc09ad/source_snapshot/overnight_run_07_12_sfm"
PY="$HOME/miniforge3/envs/cfm_mppi/bin/python"
V=/data3/research1/claude_sfm2_predictive_cfc09ad/kazuki_compare/v3
R0="$HOME/projects/safe_flow_expansion_SFM2-claude-cfc09ad/checkpoints/hp100_pretrained_r0_258999ae.pt"
EP=${EP:-920020}
export PYTHONPATH=$PWD
for WS in 1.0 1.4; do
  TAG=${WS/./p}
  CUDA_VISIBLE_DEVICES=3 "$PY" run_kazuki_ext.py \
    --checkpoint "$R0" --episodes "$EP" --gammas 0.1,0.5,1.0 \
    --safe-coef "$WS" --goal-coef 0.5 \
    --output "$V/kazuki_ws$TAG" --device cuda:0 \
    > "$V/kazuki_ws$TAG.log" 2>&1 || echo "ws $WS FAILED"
  echo "WS $WS done"
done
echo WS_SWEEP_DONE
