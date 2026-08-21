#!/usr/bin/env bash
# Night-5 Task 1: anti-conservatism distillation sweep (4 arms + M20 screens).
# Marker-gated and re-runnable. Waits for the night-4 epilogue to free the
# GPUs before starting. GPU0: N5D3, N5D4 -> screen N5a. GPU3: N5D1, N5D2 ->
# screen N5b.
set -uo pipefail
cd "$HOME/projects/safe_flow_expansion_SFM2-claude-cfc09ad"
PY="$HOME/miniforge3/envs/cfm_mppi/bin/python"
SRC=source_snapshot/overnight_run_07_12_sfm
OUT=/data3/research1/claude_sfm2_predictive_cfc09ad
N4=$OUT/night4
MK=$N4/markers
mkdir -p "$N4/logs"
stamp() { echo "[$(date -u +%FT%TZ)][n5] $*"; }

R0=checkpoints/hp100_pretrained_r0_258999ae.pt
R0_SHA=258999ae8ccee8aec5aab92a6f751221d3c15583ac26e0a7ec8311f13316ec44
RC3=$OUT/champions/RC3_12500/snapshot_step12500.pt
RC3_SHA=f0253de394cea2bbdfe31c4fe5a3fc23bac9ae8ebbc7f109369b094c45b5a748

# Do not contend with the night-4 epilogue.
for _ in $(seq 1 180); do
  [ -f "$MK/epilogue_all.done" ] && break
  sleep 60
done
[ -f "$MK/epilogue_all.done" ] || stamp "WARN: epilogue marker missing after 3h; proceeding"

MAN8=()
for a in $OUT/ext_collect/job1 $OUT/ext_collect/job2 $OUT/ext_collect/job3 \
         $OUT/ext_collect/job4 $OUT/night2/big1_a $OUT/night2/big1_b \
         $OUT/night2/big2_a $OUT/night2/big2_b; do
  MAN8+=(--raw-manifest "$a/raw_obs")
done
TAG8=()
for a in $OUT/ext_collect/job1 $OUT/ext_collect/job2 $OUT/ext_collect/job3 \
         $OUT/ext_collect/job4 $OUT/night2/big1_a $OUT/night2/big1_b \
         $OUT/night2/big2_a $OUT/night2/big2_b; do
  TAG8+=(--archive "$a/expansion_archive_r1_tagged.pt")
done
BON_R1=()
for d in "$N4"/bon_collect_gpu0/block_* "$N4"/bon_collect_gpu3/block_*; do
  [ -f "$d/BLOCK_COMPLETE.json" ] || continue
  BON_R1+=(--archive "$d/bon_archive.pt" --raw-manifest "$d/raw_obs")
done
BON_R2=()
for d in "$N4"/bon_r2_gpu0/block_* "$N4"/bon_r2_gpu3/block_*; do
  [ -f "$d/BLOCK_COMPLETE.json" ] || continue
  BON_R2+=(--archive "$d/bon_archive.pt" --raw-manifest "$d/raw_obs")
done
stamp "r1 pairs: $(( ${#BON_R1[@]} / 4 )) blocks, r2 pairs: $(( ${#BON_R2[@]} / 4 )) blocks"

train() {  # gpu name ckpt sha lr passes snap mass extra-archive-array-name
  local gpu=$1 name=$2 ckpt=$3 sha=$4 lr=$5 passes=$6 snap=$7 mass=$8
  shift 8
  local D=$OUT/ext_train/arm_$name
  [ -f "$MK/n5_${name}.done" ] && { stamp "$name already done"; return 0; }
  rm -rf "$D"
  stamp "training $name"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" $SRC/sfm_hp100_raw_train.py \
    --checkpoint "$ckpt" --expected-checkpoint-sha256 "$sha" \
    --output "$D" "$@" \
    --raw-audit-atol 10 --lru-shards 300 \
    --context-path raw --optimizer-scope all_open \
    --alpha 0.02 --negative-mode hinge --negative-margin 2.0 \
    --positive-mass "$mass" \
    --learning-rate "$lr" --batch-size 64 --exposure-passes "$passes" \
    --grad-clip-norm 1.0 --train-mode eval --update-seed 2 \
    --device cuda:0 --physical-gpu "$gpu" --snapshot-every "$snap" \
    > "$N4/logs/train_${name}.log" 2>&1
  [ -f "$D/checkpoint_r1.pt" ] \
    && { touch "$MK/n5_${name}.done"; stamp "$name done"; } \
    || stamp "$name FAILED (chain continues)"
}

screen_stage() {  # gpu stagename ck-string
  local gpu=$1 st=$2 CK=$3
  [ -f "$MK/n5_screen_${st}.done" ] && { stamp "screen $st already done"; return 0; }
  stamp "screening $st"
  OUTPUT=$OUT/funnel/screen_m20_$st STAGE=screen-m20 CHECKPOINTS="$CK" \
    PHYSICAL_GPU=$gpu PYTHON_BIN=$PY bash scripts/run_expansion_funnel.sh \
    > "$OUT/funnel/screen_m20_$st.log" 2>&1
  [ -f "$OUT/funnel/screen_m20_$st/STAGE_COMPLETE.json" ] \
    && { touch "$MK/n5_screen_${st}.done"; stamp "screen $st done"; } \
    || stamp "screen $st FAILED"
}

snaps_ck() {  # arm-name steps...
  local arm=$1; shift
  local D=$OUT/ext_train/arm_$arm CK=""
  [ -f "$D/checkpoint_r1.pt" ] && CK="$arm=$D/checkpoint_r1.pt"
  for s in "$@"; do
    [ -f "$D/snapshot_step${s}.pt" ] && CK="$CK ${arm}_s${s}=$D/snapshot_step${s}.pt"
  done
  echo "$CK"
}

(
  train 0 N5D3 "$R0" "$R0_SHA" 1e-5 4 2500 per_gamma_balanced \
    "${TAG8[@]}" "${BON_R1[@]}" "${BON_R2[@]}" --archive "$N4/negs_only.pt" "${MAN8[@]}"
  train 0 N5D4 "$RC3" "$RC3_SHA" 5e-6 2 1250 per_gamma_balanced \
    "${BON_R1[@]}" --archive "$N4/negs_only.pt" "${MAN8[@]}"
  CK="$(snaps_ck N5D3 20000 25000 30000 32500 35000 40000) $(snaps_ck N5D4 02500 05000 07500 10000)"
  screen_stage 0 N5a "$CK"
) &
(
  train 3 N5D1 "$RC3" "$RC3_SHA" 5e-6 4 1250 per_gamma_balanced \
    "${BON_R1[@]}" "${BON_R2[@]}" --archive "$N4/negs_only.pt" "${MAN8[@]}"
  train 3 N5D2 "$RC3" "$RC3_SHA" 5e-6 4 1250 progress_weighted \
    "${BON_R1[@]}" --archive "$N4/negs_only.pt" "${MAN8[@]}"
  CK="$(snaps_ck N5D1 10000 15000 17500 20000 22500 25000) $(snaps_ck N5D2 07500 10000 12500 15000)"
  screen_stage 3 N5b "$CK"
) &
wait
touch "$MK/night5_task1.done"
stamp "NIGHT5 TASK1 COMPLETE"
echo NIGHT5_TASK1_DONE
