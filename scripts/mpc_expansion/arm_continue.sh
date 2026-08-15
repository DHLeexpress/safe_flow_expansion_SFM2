#!/usr/bin/env bash
# Continue one MPC-rule arm r2..r5 from resume_r1, then M20-screen r2-r5.
# Usage: arm_continue.sh <ARM_DIR_NAME> <RECIPE_ID> <SCOPE> <E> <LABEL_PREFIX>
# Optional env: REPLAY_WINDOW (declared archive replay window, default unset
# = orchestrator default 1); MPC_LAM/MPC_RHO/MPC_R_EFF/MPC_SIGMA pass through
# to the driver.
set -uo pipefail
ARM=$1; RID=$2; SCOPE=$3; E=$4; PFX=$5
cd "$HOME/projects/safe_flow_expansion_SFM2-claude-cfc09ad"
PY="$HOME/miniforge3/envs/cfm_mppi/bin/python"
OUT=/data3/research1/claude_sfm2_predictive_cfc09ad
M=$OUT/mpc_expansion
D=$M/$ARM
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$PWD/source_snapshot/overnight_run_07_12_sfm
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }

REPLAY_ARGS=()
if [ -n "${REPLAY_WINDOW:-}" ]; then
  REPLAY_ARGS=(--replay-window "$REPLAY_WINDOW")
fi

if [ ! -f "$D/checkpoint_r5.pt" ]; then
  stamp "$RID rounds 2-5 (resume from r1)"
  "$PY" "$M/mpc_round_driver.py" \
    --checkpoint checkpoints/hp100_pretrained_r0_258999ae.pt \
    --expected-checkpoint-sha256 258999ae8ccee8aec5aab92a6f751221d3c15583ac26e0a7ec8311f13316ec44 \
    --pretrain-dataset-root /data3/research1/sfm_hp100_certified_weighted_500x7_2671a94 \
    --expected-pretrain-dataset-manifest-sha256 44f2bfa8afbb2318376ae9e188b1b622f102253a4a91f5c8ca0f9634d5041c94 \
    --output "$D" --recipe-id "$RID" --resume-from "$D/resume_r1.pt" \
    --device cuda:0 --physical-gpu 3 --verifier-workers 32 \
    --rounds 5 --alpha 0.0 --learning-rate 1e-5 --batch-size 64 \
    --exposure-passes "$E" --optimizer-scope "$SCOPE" \
    --grad-clip-norm 1.0 --max-relative-parameter-drift 0.25 \
    --train-mode eval --update-seed 2 \
    --scene-profile double_density_velocity_ood \
    --gammas 0.1,0.2,0.3,0.4,0.5,0.7,1.0 --lineages-per-gamma 8 \
    --max-steps 180 --max-attempts 32 --ess-target 0.1 --seed 41 \
    ${REPLAY_ARGS[@]+"${REPLAY_ARGS[@]}"} \
    >> "$D.log" 2>&1 || stamp "$RID CONTINUATION FAILED rc=$?"
fi
echo "ROUNDS_PHASE_DONE_$RID"

CK=""
for R in 2 3 4 5; do
  P=$D/checkpoint_r$R.pt
  [ -f "$P" ] && CK="$CK ${PFX}_r${R}=$P"
done
S=$OUT/funnel/screen_m20_${PFX}_r2r5
if [ -n "$CK" ] && [ ! -d "$S" ]; then
  stamp "$RID M20 screen of r2-r5"
  OUTPUT=$S STAGE=screen-m20 CHECKPOINTS="${CK# }" PHYSICAL_GPU=3 \
    PYTHON_BIN=$PY bash scripts/run_expansion_funnel.sh > "$S.log" 2>&1 \
    || stamp "$RID SCREEN FAILED rc=$?"
fi
echo "ARM_PIPELINE_DONE_$RID"
