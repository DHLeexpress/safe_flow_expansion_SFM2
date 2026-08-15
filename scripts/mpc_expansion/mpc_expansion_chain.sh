#!/usr/bin/env bash
# Phase D chain: three arms (M-E1, M-E4, M-E1L) trained on ONE shared
# round-1 archive gathered under the MPC-cost rule, then a joint M20 screen.
set -uo pipefail
cd "$HOME/projects/safe_flow_expansion_SFM2-claude-cfc09ad"
PY="$HOME/miniforge3/envs/cfm_mppi/bin/python"
OUT=/data3/research1/claude_sfm2_predictive_cfc09ad
M=$OUT/mpc_expansion
DRIVER=$M/mpc_round_driver.py
R0=checkpoints/hp100_pretrained_r0_258999ae.pt
R0_SHA=258999ae8ccee8aec5aab92a6f751221d3c15583ac26e0a7ec8311f13316ec44
DS=/data3/research1/sfm_hp100_certified_weighted_500x7_2671a94
DS_SHA=44f2bfa8afbb2318376ae9e188b1b622f102253a4a91f5c8ca0f9634d5041c94
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$PWD/source_snapshot/overnight_run_07_12_sfm
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }

mkdir -p "$M"
cat > "$M/MPC_EXPANSION_DECLARED.json" <<DEOF
{"declared": ["M-E1: trunk_and_head E=1", "M-E4: trunk_and_head E=4",
  "M-E1L: last_block_and_head E=1"],
 "alpha": 0.0, "lr": 1e-5, "rounds": 1,
 "acquisition_rule": "predictive_mpc_v2 lam=4 rho=1.1 r_eff=0.45 sigma=0.1",
 "shared_round1_archive": true,
 "screen": "same fixed CRN M20 bank as contract v3 (OOD ep0 900000, seed 20260814)",
 "note": "declared before any M50/M100 read; formal contract amendment to follow in-repo"}
DEOF

run_arm() {
  local id=$1 scope=$2 E=$3 reuse=$4
  local D=$M/arm_$id
  [ -f "$D/checkpoint_r1.pt" ] && { stamp "arm $id already complete"; return 0; }
  stamp "arm $id (scope=$scope E=$E reuse=${reuse:-none})"
  local extra=()
  [ -n "$reuse" ] && extra=(--reuse-archive "$reuse")
  "$PY" "$DRIVER" \
    --checkpoint "$R0" --expected-checkpoint-sha256 "$R0_SHA" \
    --pretrain-dataset-root "$DS" \
    --expected-pretrain-dataset-manifest-sha256 "$DS_SHA" \
    --output "$D" --recipe-id "$id" \
    --device cuda:0 --physical-gpu 3 --verifier-workers 32 \
    --rounds 1 --alpha 0.0 --learning-rate 1e-5 --batch-size 64 \
    --exposure-passes "$E" --optimizer-scope "$scope" \
    --grad-clip-norm 1.0 --max-relative-parameter-drift 0.25 \
    --train-mode eval --update-seed 2 \
    --scene-profile double_density_velocity_ood \
    --gammas 0.1,0.2,0.3,0.4,0.5,0.7,1.0 --lineages-per-gamma 8 \
    --max-steps 180 --max-attempts 32 --ess-target 0.1 --seed 41 \
    "${extra[@]}" > "$D.log" 2>&1 || stamp "ARM $id FAILED rc=$?"
}

run_arm M-E1 trunk_and_head 1 ""
ARCHIVE=$(ls $M/arm_M-E1/round_01/expansion_archive_r1.pt 2>/dev/null || true)
if [ -z "$ARCHIVE" ]; then
  ARCHIVE=$(find $M/arm_M-E1 -name "expansion_archive_r1.pt" | head -1)
fi
if [ -z "$ARCHIVE" ]; then
  stamp "no shared archive from M-E1; aborting sibling arms"
  echo MPC_EXPANSION_CHAIN_DONE; exit 1
fi
stamp "shared archive: $ARCHIVE"
run_arm M-E4 trunk_and_head 4 "$ARCHIVE"
run_arm M-E1L last_block_and_head 1 "$ARCHIVE"

stamp "joint M20 screen of the new checkpoints"
CK=""
for id in M-E1 M-E4 M-E1L; do
  P=$M/arm_$id/checkpoint_r1.pt
  [ -f "$P" ] && CK="$CK ${id//-/}_r1=$P"
done
if [ -n "$CK" ] && [ ! -d "$OUT/funnel/screen_m20_mpc" ]; then
  OUTPUT=$OUT/funnel/screen_m20_mpc STAGE=screen-m20 CHECKPOINTS="${CK# }" \
    PHYSICAL_GPU=3 PYTHON_BIN=$PY bash scripts/run_expansion_funnel.sh \
    > "$OUT/funnel/screen_m20_mpc.log" 2>&1 || stamp "SCREEN FAILED rc=$?"
fi
stamp "chain finished"
echo MPC_EXPANSION_CHAIN_DONE
