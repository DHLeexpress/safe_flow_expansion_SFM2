#!/usr/bin/env bash
# Night-5 Task 2: BoN-32 teacher collection (RC3 teacher, sharper selection)
# on both GPUs, then student N5E1 (RC3-continue on BoN-32 only) on GPU3 and
# its M20 screen on GPU0. Marker-gated; waits for Task 1 to free the GPUs.
set -uo pipefail
cd "$HOME/projects/safe_flow_expansion_SFM2-claude-cfc09ad"
PY="$HOME/miniforge3/envs/cfm_mppi/bin/python"
SRC=source_snapshot/overnight_run_07_12_sfm
OUT=/data3/research1/claude_sfm2_predictive_cfc09ad
N4=$OUT/night4
MK=$N4/markers
mkdir -p "$N4/logs"
stamp() { echo "[$(date -u +%FT%TZ)][n5b] $*"; }

RC3=$OUT/champions/RC3_12500/snapshot_step12500.pt
RC3_SHA=f0253de394cea2bbdfe31c4fe5a3fc23bac9ae8ebbc7f109369b094c45b5a748

for _ in $(seq 1 360); do
  [ -f "$MK/night5_task1.done" ] && break
  sleep 60
done
[ -f "$MK/night5_task1.done" ] || stamp "WARN: task1 marker missing after 6h; proceeding"

DEADLINE=$(( $(date +%s) + 120*60 ))
stamp "BoN-32 collect deadline epoch $DEADLINE"

collect() {  # gpu ep0 baseseed extraseed outdir
  CUDA_VISIBLE_DEVICES=$1 "$PY" $SRC/sfm_hp100_bon_distill_collect.py \
    --checkpoint "$RC3" --expected-checkpoint-sha256 "$RC3_SHA" \
    --label BON32_RC3_gpu$1 --scene-profile double_density_velocity_ood \
    --ep0-start $2 --block-stride 1000 --blocks 40 --M 20 --N 32 \
    --base-seed-start $3 --extra-seed-start $4 \
    --deadline-epoch "$DEADLINE" --verify-workers 8 \
    --out-dir "$5" --device cuda:0 --physical-gpu $1 \
    > "$N4/logs/collect_bon32_gpu$1.log" 2>&1
}

if [ ! -f "$MK/n5_bon32_collect.done" ]; then
  ( collect 0 840000 39000000 40000000 "$N4/bon32_gpu0" ) &
  ( collect 3 890000 41000000 42000000 "$N4/bon32_gpu3" ) &
  wait
  if [ -f "$N4/bon32_gpu0/COLLECT_COMPLETE.json" ] \
     || [ -f "$N4/bon32_gpu3/COLLECT_COMPLETE.json" ]; then
    touch "$MK/n5_bon32_collect.done"; stamp "BoN-32 collect done"
  else
    stamp "BoN-32 collect FAILED on both GPUs"; exit 1
  fi
else
  stamp "BoN-32 collect already done"
fi

if [ ! -f "$MK/n5_N5E1.done" ]; then
  MAN8=()
  for a in $OUT/ext_collect/job1 $OUT/ext_collect/job2 $OUT/ext_collect/job3 \
           $OUT/ext_collect/job4 $OUT/night2/big1_a $OUT/night2/big1_b \
           $OUT/night2/big2_a $OUT/night2/big2_b; do
    MAN8+=(--raw-manifest "$a/raw_obs")
  done
  ARCH=(--archive "$N4/negs_only.pt" "${MAN8[@]}")
  nblocks=0
  for d in "$N4"/bon32_gpu0/block_* "$N4"/bon32_gpu3/block_*; do
    [ -f "$d/BLOCK_COMPLETE.json" ] || continue
    ARCH+=(--archive "$d/bon_archive.pt" --raw-manifest "$d/raw_obs")
    nblocks=$((nblocks+1))
  done
  stamp "training N5E1 with $nblocks BoN-32 blocks"
  if [ "$nblocks" -lt 2 ]; then stamp "FATAL: <2 blocks"; exit 1; fi
  D=$OUT/ext_train/arm_N5E1
  rm -rf "$D"
  CUDA_VISIBLE_DEVICES=3 "$PY" $SRC/sfm_hp100_raw_train.py \
    --checkpoint "$RC3" --expected-checkpoint-sha256 "$RC3_SHA" \
    --output "$D" "${ARCH[@]}" \
    --raw-audit-atol 10 --lru-shards 250 \
    --context-path raw --optimizer-scope all_open \
    --alpha 0.02 --negative-mode hinge --negative-margin 2.0 \
    --positive-mass per_gamma_balanced \
    --learning-rate 5e-6 --batch-size 64 --exposure-passes 4 \
    --grad-clip-norm 1.0 --train-mode eval --update-seed 2 \
    --device cuda:0 --physical-gpu 3 --snapshot-every 1250 \
    > "$N4/logs/train_N5E1.log" 2>&1
  [ -f "$D/checkpoint_r1.pt" ] \
    && { touch "$MK/n5_N5E1.done"; stamp "N5E1 done"; } \
    || { stamp "N5E1 FAILED"; exit 1; }
fi

if [ ! -f "$MK/n5_screen_N5c.done" ]; then
  D=$OUT/ext_train/arm_N5E1
  CK="N5E1=$D/checkpoint_r1.pt"
  for s in 02500 05000 07500 10000 12500; do
    [ -f "$D/snapshot_step${s}.pt" ] && CK="$CK N5E1_s${s}=$D/snapshot_step${s}.pt"
  done
  stamp "screening N5c: $CK"
  OUTPUT=$OUT/funnel/screen_m20_N5c STAGE=screen-m20 CHECKPOINTS="$CK" \
    PHYSICAL_GPU=0 PYTHON_BIN=$PY bash scripts/run_expansion_funnel.sh \
    > "$OUT/funnel/screen_m20_N5c.log" 2>&1
  [ -f "$OUT/funnel/screen_m20_N5c/STAGE_COMPLETE.json" ] \
    && { touch "$MK/n5_screen_N5c.done"; stamp "screen N5c done"; } \
    || { stamp "screen N5c FAILED"; exit 1; }
fi

touch "$MK/night5_task2.done"
stamp "NIGHT5 TASK2 COMPLETE"
echo NIGHT5_TASK2_DONE
