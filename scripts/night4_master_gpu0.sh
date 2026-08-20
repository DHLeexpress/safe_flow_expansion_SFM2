#!/usr/bin/env bash
# Night-4 offensive, GPU 0 chain:
#   [collect] BoN-16 RC3-teacher distillation blocks (ep0 800000+)
#   [train]   N4A1 = r0 + 8 tagged archives + all BoN blocks (champion profile)
#   [screen]  M20 on N4A1 final + snapshot grid
# Stage markers under $OUT/night4/markers; every stage skips if its marker
# exists, so the script is re-runnable after a crash.
set -uo pipefail
cd "$HOME/projects/safe_flow_expansion_SFM2-claude-cfc09ad"
PY="$HOME/miniforge3/envs/cfm_mppi/bin/python"
SRC=source_snapshot/overnight_run_07_12_sfm
OUT=/data3/research1/claude_sfm2_predictive_cfc09ad
N4=$OUT/night4
MK=$N4/markers
mkdir -p "$N4" "$MK" "$N4/logs"
stamp() { echo "[$(date -u +%FT%TZ)][gpu0] $*"; }

R0=checkpoints/hp100_pretrained_r0_258999ae.pt
R0_SHA=258999ae8ccee8aec5aab92a6f751221d3c15583ac26e0a7ec8311f13316ec44
RC3=$OUT/champions/RC3_12500/snapshot_step12500.pt
RC3_SHA=f0253de394cea2bbdfe31c4fe5a3fc23bac9ae8ebbc7f109369b094c45b5a748

# Shared wall-clock deadline for BOTH GPUs' collect stages (first writer wins).
if [ ! -f "$N4/DEADLINE_EPOCH" ]; then
  echo $(( $(date +%s) + 230*60 )) > "$N4/DEADLINE_EPOCH".tmp \
    && mv -n "$N4/DEADLINE_EPOCH".tmp "$N4/DEADLINE_EPOCH" || true
fi
DEADLINE=$(cat "$N4/DEADLINE_EPOCH")
stamp "deadline epoch $DEADLINE"

# ---------------- stage: collect (GPU 0) ----------------
if [ ! -f "$MK/collect0.done" ]; then
  stamp "collect0 start"
  CUDA_VISIBLE_DEVICES=0 "$PY" $SRC/sfm_hp100_bon_distill_collect.py \
    --checkpoint "$RC3" --expected-checkpoint-sha256 "$RC3_SHA" \
    --label BON16_RC3_gpu0 --scene-profile double_density_velocity_ood \
    --ep0-start 800000 --block-stride 1000 --blocks 40 --M 20 --N 16 \
    --base-seed-start 31000000 --extra-seed-start 32000000 \
    --deadline-epoch "$DEADLINE" --verify-workers 8 \
    --out-dir "$N4/bon_collect_gpu0" --device cuda:0 --physical-gpu 0 \
    > "$N4/logs/collect_gpu0.log" 2>&1
  rc=$?
  if [ -f "$N4/bon_collect_gpu0/COLLECT_COMPLETE.json" ]; then
    touch "$MK/collect0.done"; stamp "collect0 done rc=$rc"
  else
    stamp "collect0 FAILED rc=$rc (no COLLECT_COMPLETE)"; exit 1
  fi
else
  stamp "collect0 already done"
fi

# ---------------- wait for GPU 3 collect ----------------
stamp "waiting for collect3.done"
for _ in $(seq 1 90); do
  [ -f "$MK/collect3.done" ] && break
  sleep 60
done
[ -f "$MK/collect3.done" ] || stamp "WARN: collect3 marker missing after 90min; training on gpu0 blocks only"

# ---------------- stage: train N4A1 ----------------
if [ ! -f "$MK/a1.done" ]; then
  ARCH=()
  for a in $OUT/ext_collect/job1 $OUT/ext_collect/job2 $OUT/ext_collect/job3 \
           $OUT/ext_collect/job4 $OUT/night2/big1_a $OUT/night2/big1_b \
           $OUT/night2/big2_a $OUT/night2/big2_b; do
    ARCH+=(--archive "$a/expansion_archive_r1_tagged.pt" --raw-manifest "$a/raw_obs")
  done
  nblocks=0
  for d in "$N4"/bon_collect_gpu0/block_* "$N4"/bon_collect_gpu3/block_*; do
    [ -f "$d/BLOCK_COMPLETE.json" ] || continue
    ARCH+=(--archive "$d/bon_archive.pt" --raw-manifest "$d/raw_obs")
    nblocks=$((nblocks+1))
  done
  stamp "training N4A1 with $nblocks BoN blocks"
  if [ "$nblocks" -lt 2 ]; then stamp "FATAL: <2 BoN blocks"; exit 1; fi
  D=$OUT/ext_train/arm_N4A1
  rm -rf "$D"
  CUDA_VISIBLE_DEVICES=0 "$PY" $SRC/sfm_hp100_raw_train.py \
    --checkpoint "$R0" --expected-checkpoint-sha256 "$R0_SHA" \
    --output "$D" "${ARCH[@]}" \
    --raw-audit-atol 10 --lru-shards 350 \
    --context-path raw --optimizer-scope all_open \
    --alpha 0.02 --negative-mode hinge --negative-margin 2.0 \
    --positive-mass per_gamma_balanced \
    --learning-rate 1e-5 --batch-size 64 --exposure-passes 4 \
    --grad-clip-norm 1.0 --train-mode eval --update-seed 2 \
    --device cuda:0 --physical-gpu 0 --snapshot-every 2500 \
    > "$N4/logs/train_N4A1.log" 2>&1
  rc=$?
  if [ -f "$D/checkpoint_r1.pt" ]; then
    touch "$MK/a1.done"; stamp "N4A1 done rc=$rc"
  else
    stamp "N4A1 FAILED rc=$rc"; exit 1
  fi
else
  stamp "N4A1 already done"
fi

# ---------------- stage: screen N4a ----------------
if [ ! -f "$MK/screen_n4a.done" ]; then
  D=$OUT/ext_train/arm_N4A1
  CK="N4A1=$D/checkpoint_r1.pt"
  for s in 07500 10000 12500 15000 17500 20000 22500 25000 27500 30000; do
    P=$D/snapshot_step${s}.pt
    [ -f "$P" ] && CK="$CK N4A1_s${s}=$P"
  done
  stamp "screening N4a: $CK"
  OUTPUT=$OUT/funnel/screen_m20_N4a STAGE=screen-m20 CHECKPOINTS="$CK" \
    PHYSICAL_GPU=0 PYTHON_BIN=$PY bash scripts/run_expansion_funnel.sh \
    > "$OUT/funnel/screen_m20_N4a.log" 2>&1
  if [ -f "$OUT/funnel/screen_m20_N4a/STAGE_COMPLETE.json" ]; then
    touch "$MK/screen_n4a.done"; stamp "screen N4a done"
  else
    stamp "screen N4a FAILED"; exit 1
  fi
else
  stamp "screen N4a already done"
fi

touch "$MK/gpu0_all.done"
stamp "GPU0 CHAIN COMPLETE"
echo NIGHT4_GPU0_DONE
