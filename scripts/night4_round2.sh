#!/usr/bin/env bash
# Night-4 round 2: iterate the BoN-16 self-distillation with the NEW champion
# as teacher (DAgger round).  Both GPUs collect, GPU3 trains N4A3
# (N4A2_s12500-continue on round-2 BoN only), GPU0 screens M20.
set -uo pipefail
cd "$HOME/projects/safe_flow_expansion_SFM2-claude-cfc09ad"
PY="$HOME/miniforge3/envs/cfm_mppi/bin/python"
SRC=source_snapshot/overnight_run_07_12_sfm
OUT=/data3/research1/claude_sfm2_predictive_cfc09ad
N4=$OUT/night4
MK=$N4/markers
mkdir -p "$N4/logs"
stamp() { echo "[$(date -u +%FT%TZ)][r2] $*"; }

A2=$OUT/champions/N4A2_12500/snapshot_step12500.pt
A2_SHA=bb1c3e1544ccc4c9da8c40caeb3762047fb25f78bb3a327fe5b64500bf2e76f8
DEADLINE=$(( $(date +%s) + 85*60 ))
stamp "round-2 collect deadline epoch $DEADLINE"

collect() {  # gpu ep0 baseseed extraseed outdir
  CUDA_VISIBLE_DEVICES=$1 "$PY" $SRC/sfm_hp100_bon_distill_collect.py \
    --checkpoint "$A2" --expected-checkpoint-sha256 "$A2_SHA" \
    --label BON16_A2_gpu$1 --scene-profile double_density_velocity_ood \
    --ep0-start $2 --block-stride 1000 --blocks 40 --M 20 --N 16 \
    --base-seed-start $3 --extra-seed-start $4 \
    --deadline-epoch "$DEADLINE" --verify-workers 8 \
    --out-dir "$5" --device cuda:0 --physical-gpu $1 \
    > "$N4/logs/collect_r2_gpu$1.log" 2>&1
}

if [ ! -f "$MK/r2_collect.done" ]; then
  ( collect 0 820000 35000000 36000000 "$N4/bon_r2_gpu0" ) &
  ( collect 3 880000 37000000 38000000 "$N4/bon_r2_gpu3" ) &
  wait
  if [ -f "$N4/bon_r2_gpu0/COLLECT_COMPLETE.json" ] \
     || [ -f "$N4/bon_r2_gpu3/COLLECT_COMPLETE.json" ]; then
    touch "$MK/r2_collect.done"; stamp "round-2 collect done"
  else
    stamp "round-2 collect FAILED on both GPUs"; exit 1
  fi
else
  stamp "round-2 collect already done"
fi

if [ ! -f "$MK/r2_a3.done" ]; then
  MAN8=()
  for a in $OUT/ext_collect/job1 $OUT/ext_collect/job2 $OUT/ext_collect/job3 \
           $OUT/ext_collect/job4 $OUT/night2/big1_a $OUT/night2/big1_b \
           $OUT/night2/big2_a $OUT/night2/big2_b; do
    MAN8+=(--raw-manifest "$a/raw_obs")
  done
  ARCH=(--archive "$N4/negs_only.pt" "${MAN8[@]}")
  nblocks=0
  for d in "$N4"/bon_r2_gpu0/block_* "$N4"/bon_r2_gpu3/block_*; do
    [ -f "$d/BLOCK_COMPLETE.json" ] || continue
    ARCH+=(--archive "$d/bon_archive.pt" --raw-manifest "$d/raw_obs")
    nblocks=$((nblocks+1))
  done
  stamp "training N4A3 with $nblocks round-2 blocks"
  if [ "$nblocks" -lt 2 ]; then stamp "FATAL: <2 blocks"; exit 1; fi
  D=$OUT/ext_train/arm_N4A3
  rm -rf "$D"
  CUDA_VISIBLE_DEVICES=3 "$PY" $SRC/sfm_hp100_raw_train.py \
    --checkpoint "$A2" --expected-checkpoint-sha256 "$A2_SHA" \
    --output "$D" "${ARCH[@]}" \
    --raw-audit-atol 10 --lru-shards 250 \
    --context-path raw --optimizer-scope all_open \
    --alpha 0.02 --negative-mode hinge --negative-margin 2.0 \
    --positive-mass per_gamma_balanced \
    --learning-rate 5e-6 --batch-size 64 --exposure-passes 4 \
    --grad-clip-norm 1.0 --train-mode eval --update-seed 2 \
    --device cuda:0 --physical-gpu 3 --snapshot-every 1250 \
    > "$N4/logs/train_N4A3.log" 2>&1
  if [ -f "$D/checkpoint_r1.pt" ]; then
    touch "$MK/r2_a3.done"; stamp "N4A3 done"
  else
    stamp "N4A3 FAILED"; exit 1
  fi
else
  stamp "N4A3 already done"
fi

if [ ! -f "$MK/r2_screen.done" ]; then
  D=$OUT/ext_train/arm_N4A3
  CK="N4A3=$D/checkpoint_r1.pt"
  for s in 02500 05000 07500 10000 12500; do
    P=$D/snapshot_step${s}.pt
    [ -f "$P" ] && CK="$CK N4A3_s${s}=$P"
  done
  stamp "screening N4c: $CK"
  OUTPUT=$OUT/funnel/screen_m20_N4c STAGE=screen-m20 CHECKPOINTS="$CK" \
    PHYSICAL_GPU=0 PYTHON_BIN=$PY bash scripts/run_expansion_funnel.sh \
    > "$OUT/funnel/screen_m20_N4c.log" 2>&1
  if [ -f "$OUT/funnel/screen_m20_N4c/STAGE_COMPLETE.json" ]; then
    touch "$MK/r2_screen.done"; stamp "screen N4c done"
  else
    stamp "screen N4c FAILED"; exit 1
  fi
fi

touch "$MK/round2_all.done"
stamp "ROUND 2 COMPLETE"
echo NIGHT4_ROUND2_DONE
