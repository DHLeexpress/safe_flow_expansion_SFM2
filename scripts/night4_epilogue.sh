#!/usr/bin/env bash
# Night-4 epilogue: (GPU0) best-of-N calibration of the new champion on the
# same M20 bank/seeds as the prior sweep; (GPU3) update-seed-3 replication of
# the A2 recipe + M20 screen of its s12500/final points.
set -uo pipefail
cd "$HOME/projects/safe_flow_expansion_SFM2-claude-cfc09ad"
PY="$HOME/miniforge3/envs/cfm_mppi/bin/python"
SRC=source_snapshot/overnight_run_07_12_sfm
OUT=/data3/research1/claude_sfm2_predictive_cfc09ad
N4=$OUT/night4
MK=$N4/markers
DIR=$OUT/bestofN_calibration
stamp() { echo "[$(date -u +%FT%TZ)][epi] $*"; }

A2=$OUT/champions/N4A2_12500/snapshot_step12500.pt
A2_SHA=bb1c3e1544ccc4c9da8c40caeb3762047fb25f78bb3a327fe5b64500bf2e76f8
RC3=$OUT/champions/RC3_12500/snapshot_step12500.pt
RC3_SHA=f0253de394cea2bbdfe31c4fe5a3fc23bac9ae8ebbc7f109369b094c45b5a748

# ---- GPU0: BoN calibration for N4A2_12500 (mirrors run_bestofN.sh) ----
bon_calib() {
  local shard=$1 grid=$2 flags=${3:-}
  CUDA_VISIBLE_DEVICES=0 "$PY" -u "$DIR/bestofN_calibration.py" \
    --checkpoint "$A2" --label N4A2_12500 --N-grid "$grid" \
    --scene-profile double_density_velocity_ood --ep0 900000 --M 20 \
    --base-seed 20260814 --extra-seed 20260819 --device cuda \
    --self-check-steps 2 $flags \
    --out "$DIR/N4A2_12500_shard${shard}.json" \
    > "$DIR/logs/N4A2_12500_shard${shard}.log" 2>&1
}
(
  if [ ! -f "$MK/epi_calib.done" ]; then
    stamp "BoN calibration shards A+B for N4A2_12500"
    bon_calib A "1,2,4" "--prove-n1"
    bon_calib B "8,16" ""
    if [ -f "$DIR/N4A2_12500_shardA.json" ] && [ -f "$DIR/N4A2_12500_shardB.json" ]; then
      touch "$MK/epi_calib.done"; stamp "calibration done"
    else
      stamp "calibration FAILED"
    fi
  fi
) &

# ---- GPU3: seed-3 replication of the A2 recipe ----
(
  if [ ! -f "$MK/epi_rep.done" ]; then
    MAN8=()
    for a in $OUT/ext_collect/job1 $OUT/ext_collect/job2 $OUT/ext_collect/job3 \
             $OUT/ext_collect/job4 $OUT/night2/big1_a $OUT/night2/big1_b \
             $OUT/night2/big2_a $OUT/night2/big2_b; do
      MAN8+=(--raw-manifest "$a/raw_obs")
    done
    ARCH=(--archive "$N4/negs_only.pt" "${MAN8[@]}")
    for d in "$N4"/bon_collect_gpu0/block_* "$N4"/bon_collect_gpu3/block_*; do
      [ -f "$d/BLOCK_COMPLETE.json" ] || continue
      ARCH+=(--archive "$d/bon_archive.pt" --raw-manifest "$d/raw_obs")
    done
    D=$OUT/ext_train/arm_N4A2_REP
    rm -rf "$D"
    stamp "training N4A2_REP (update-seed 3)"
    CUDA_VISIBLE_DEVICES=3 "$PY" $SRC/sfm_hp100_raw_train.py \
      --checkpoint "$RC3" --expected-checkpoint-sha256 "$RC3_SHA" \
      --output "$D" "${ARCH[@]}" \
      --raw-audit-atol 10 --lru-shards 250 \
      --context-path raw --optimizer-scope all_open \
      --alpha 0.02 --negative-mode hinge --negative-margin 2.0 \
      --positive-mass per_gamma_balanced \
      --learning-rate 5e-6 --batch-size 64 --exposure-passes 4 \
      --grad-clip-norm 1.0 --train-mode eval --update-seed 3 \
      --device cuda:0 --physical-gpu 3 --snapshot-every 1250 \
      > "$N4/logs/train_N4A2_REP.log" 2>&1
    if [ -f "$D/checkpoint_r1.pt" ]; then
      CK="N4A2R=$D/checkpoint_r1.pt"
      [ -f "$D/snapshot_step12500.pt" ] && CK="$CK N4A2R_s12500=$D/snapshot_step12500.pt"
      [ -f "$D/snapshot_step10000.pt" ] && CK="$CK N4A2R_s10000=$D/snapshot_step10000.pt"
      stamp "screening replication: $CK"
      OUTPUT=$OUT/funnel/screen_m20_N4rep STAGE=screen-m20 CHECKPOINTS="$CK" \
        PHYSICAL_GPU=3 PYTHON_BIN=$PY bash scripts/run_expansion_funnel.sh \
        > "$OUT/funnel/screen_m20_N4rep.log" 2>&1
      [ -f "$OUT/funnel/screen_m20_N4rep/STAGE_COMPLETE.json" ] \
        && { touch "$MK/epi_rep.done"; stamp "replication screened"; } \
        || stamp "replication screen FAILED"
    else
      stamp "N4A2_REP train FAILED"
    fi
  fi
) &
wait
touch "$MK/epilogue_all.done"
stamp "EPILOGUE COMPLETE"
echo NIGHT4_EPILOGUE_DONE
