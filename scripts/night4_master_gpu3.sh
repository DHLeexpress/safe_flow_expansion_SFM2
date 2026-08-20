#!/usr/bin/env bash
# Night-4 offensive, GPU 3 chain:
#   [trainB]  N4B1/N4B2 = RC3-continue on hard-mode data, N4B3 control on avoid
#   [trainC]  N4C1/N4C2 = DPO-v2 tuned volley (displacement diag cleared it)
#   [collect] BoN-16 RC3-teacher distillation blocks (ep0 860000+)
#   [prep]    negatives-only archive from night3/mode_hard.pt
#   [trainA2] RC3-continue on BoN blocks + negatives (self-distillation)
#   [screen]  M20 on A2 + B + C finals/snapshots
set -uo pipefail
cd "$HOME/projects/safe_flow_expansion_SFM2-claude-cfc09ad"
PY="$HOME/miniforge3/envs/cfm_mppi/bin/python"
SRC=source_snapshot/overnight_run_07_12_sfm
OUT=/data3/research1/claude_sfm2_predictive_cfc09ad
N4=$OUT/night4
MK=$N4/markers
mkdir -p "$N4" "$MK" "$N4/logs"
stamp() { echo "[$(date -u +%FT%TZ)][gpu3] $*"; }

R0=checkpoints/hp100_pretrained_r0_258999ae.pt
R0_SHA=258999ae8ccee8aec5aab92a6f751221d3c15583ac26e0a7ec8311f13316ec44
RC3=$OUT/champions/RC3_12500/snapshot_step12500.pt
RC3_SHA=f0253de394cea2bbdfe31c4fe5a3fc23bac9ae8ebbc7f109369b094c45b5a748

if [ ! -f "$N4/DEADLINE_EPOCH" ]; then
  echo $(( $(date +%s) + 230*60 )) > "$N4/DEADLINE_EPOCH".tmp \
    && mv -n "$N4/DEADLINE_EPOCH".tmp "$N4/DEADLINE_EPOCH" || true
fi
DEADLINE=$(cat "$N4/DEADLINE_EPOCH")
stamp "deadline epoch $DEADLINE"

MAN8=()
for a in $OUT/ext_collect/job1 $OUT/ext_collect/job2 $OUT/ext_collect/job3 \
         $OUT/ext_collect/job4 $OUT/night2/big1_a $OUT/night2/big1_b \
         $OUT/night2/big2_a $OUT/night2/big2_b; do
  MAN8+=(--raw-manifest "$a/raw_obs")
done

run_continue_arm() {  # name archive lr passes snapevery
  local name=$1 arch=$2 lr=$3 passes=$4 snap=$5
  local D=$OUT/ext_train/arm_$name
  [ -f "$MK/${name}.done" ] && { stamp "$name already done"; return 0; }
  rm -rf "$D"
  stamp "training $name (RC3-continue, lr=$lr E=$passes)"
  CUDA_VISIBLE_DEVICES=3 "$PY" $SRC/sfm_hp100_raw_train.py \
    --checkpoint "$RC3" --expected-checkpoint-sha256 "$RC3_SHA" \
    --output "$D" --archive "$arch" "${MAN8[@]}" \
    --raw-audit-atol 10 --lru-shards 200 \
    --context-path raw --optimizer-scope all_open \
    --alpha 0.02 --negative-mode hinge --negative-margin 2.0 \
    --positive-mass per_gamma_balanced \
    --learning-rate "$lr" --batch-size 64 --exposure-passes "$passes" \
    --grad-clip-norm 1.0 --train-mode eval --update-seed 2 \
    --device cuda:0 --physical-gpu 3 --snapshot-every "$snap" \
    > "$N4/logs/train_${name}.log" 2>&1
  if [ -f "$D/checkpoint_r1.pt" ]; then
    touch "$MK/${name}.done"; stamp "$name done"
  else
    stamp "$name FAILED (continuing chain)"
  fi
}

run_dpo_arm() {  # name beta anchor lr passes
  local name=$1 beta=$2 anchor=$3 lr=$4 passes=$5
  local D=$OUT/ext_train/arm_$name
  [ -f "$MK/${name}.done" ] && { stamp "$name already done"; return 0; }
  rm -rf "$D"
  stamp "training $name (DPO beta=$beta anchor=$anchor lr=$lr E=$passes)"
  CUDA_VISIBLE_DEVICES=3 "$PY" $SRC/sfm_hp100_dpo_train.py \
    --checkpoint "$R0" --expected-checkpoint-sha256 "$R0_SHA" \
    --reference-checkpoint "$R0" \
    --expected-reference-checkpoint-sha256 "$R0_SHA" \
    --pairs "$OUT/dpo/pairs_all.pt" "${MAN8[@]}" \
    --context-path raw --optimizer-scope trunk_head_and_projection \
    --beta "$beta" --anchor-weight "$anchor" \
    --learning-rate "$lr" --batch-size 64 --exposure-passes "$passes" \
    --grad-clip-norm 1.0 --update-seed 2 --snapshot-every 1000 \
    --lru-shards 200 --raw-audit-atol 0.005 \
    --output "$D" --device cuda:0 --physical-gpu 3 \
    > "$N4/logs/train_${name}.log" 2>&1
  if [ -f "$D/checkpoint_r1.pt" ]; then
    touch "$MK/${name}.done"; stamp "$name done"
  else
    stamp "$name FAILED (continuing chain)"
  fi
}

# ---------------- stage: B + C trainings ----------------
run_continue_arm N4B1 "$OUT/night3/mode_hard.pt"  5e-6 2 1250
run_continue_arm N4B2 "$OUT/night3/mode_hard.pt"  1e-5 4 2500
run_continue_arm N4B3 "$OUT/night3/mode_avoid.pt" 5e-6 2 1250
run_dpo_arm      N4C1 0.3 0.25 1e-5 2
run_dpo_arm      N4C2 1.0 0.10 2e-5 2

# ---------------- stage: collect (GPU 3) ----------------
if [ ! -f "$MK/collect3.done" ]; then
  stamp "collect3 start"
  CUDA_VISIBLE_DEVICES=3 "$PY" $SRC/sfm_hp100_bon_distill_collect.py \
    --checkpoint "$RC3" --expected-checkpoint-sha256 "$RC3_SHA" \
    --label BON16_RC3_gpu3 --scene-profile double_density_velocity_ood \
    --ep0-start 860000 --block-stride 1000 --blocks 40 --M 20 --N 16 \
    --base-seed-start 33000000 --extra-seed-start 34000000 \
    --deadline-epoch "$DEADLINE" --verify-workers 8 \
    --out-dir "$N4/bon_collect_gpu3" --device cuda:0 --physical-gpu 3 \
    > "$N4/logs/collect_gpu3.log" 2>&1
  rc=$?
  if [ -f "$N4/bon_collect_gpu3/COLLECT_COMPLETE.json" ]; then
    touch "$MK/collect3.done"; stamp "collect3 done rc=$rc"
  else
    stamp "collect3 FAILED rc=$rc"; exit 1
  fi
else
  stamp "collect3 already done"
fi

# ---------------- wait for GPU 0 collect ----------------
stamp "waiting for collect0.done"
for _ in $(seq 1 90); do
  [ -f "$MK/collect0.done" ] && break
  sleep 60
done
[ -f "$MK/collect0.done" ] || stamp "WARN: collect0 marker missing after 90min; A2 trains on gpu3 blocks only"

# ---------------- stage: negatives-only archive ----------------
if [ ! -f "$N4/negs_only.pt" ]; then
  stamp "building negs_only.pt"
  "$PY" - <<'EOF'
import torch
src = "/data3/research1/claude_sfm2_predictive_cfc09ad/night3/mode_hard.pt"
dst = "/data3/research1/claude_sfm2_predictive_cfc09ad/night4/negs_only.pt"
payload = torch.load(src, map_location="cpu", weights_only=False)
rows = [r for r in payload["rows"] if r["role"] != "positive"]
out = {k: v for k, v in payload.items() if k != "rows"}
out["rows"] = rows
out["negs_only"] = {"source": src, "negatives": len(rows)}
torch.save(out, dst)
print("negs_only rows:", len(rows))
EOF
fi

# ---------------- stage: train N4A2 ----------------
if [ ! -f "$MK/a2.done" ]; then
  ARCH=(--archive "$N4/negs_only.pt" "${MAN8[@]}")
  nblocks=0
  for d in "$N4"/bon_collect_gpu0/block_* "$N4"/bon_collect_gpu3/block_*; do
    [ -f "$d/BLOCK_COMPLETE.json" ] || continue
    ARCH+=(--archive "$d/bon_archive.pt" --raw-manifest "$d/raw_obs")
    nblocks=$((nblocks+1))
  done
  stamp "training N4A2 with $nblocks BoN blocks"
  if [ "$nblocks" -lt 2 ]; then stamp "FATAL: <2 BoN blocks"; exit 1; fi
  D=$OUT/ext_train/arm_N4A2
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
    > "$N4/logs/train_N4A2.log" 2>&1
  rc=$?
  if [ -f "$D/checkpoint_r1.pt" ]; then
    touch "$MK/a2.done"; stamp "N4A2 done rc=$rc"
  else
    stamp "N4A2 FAILED rc=$rc"; exit 1
  fi
else
  stamp "N4A2 already done"
fi

# ---------------- stage: screen N4b ----------------
if [ ! -f "$MK/screen_n4b.done" ]; then
  CK=""
  D=$OUT/ext_train/arm_N4A2
  [ -f "$D/checkpoint_r1.pt" ] && CK="N4A2=$D/checkpoint_r1.pt"
  for s in 02500 05000 07500 10000 12500; do
    P=$D/snapshot_step${s}.pt
    [ -f "$P" ] && CK="$CK N4A2_s${s}=$P"
  done
  for a in N4B1 N4B2 N4B3; do
    P=$OUT/ext_train/arm_$a/checkpoint_r1.pt
    [ -f "$P" ] && CK="$CK ${a}=$P"
  done
  [ -f "$OUT/ext_train/arm_N4B1/snapshot_step02500.pt" ] \
    && CK="$CK N4B1_s02500=$OUT/ext_train/arm_N4B1/snapshot_step02500.pt"
  [ -f "$OUT/ext_train/arm_N4B2/snapshot_step05000.pt" ] \
    && CK="$CK N4B2_s05000=$OUT/ext_train/arm_N4B2/snapshot_step05000.pt"
  [ -f "$OUT/ext_train/arm_N4B3/snapshot_step02500.pt" ] \
    && CK="$CK N4B3_s02500=$OUT/ext_train/arm_N4B3/snapshot_step02500.pt"
  for a in N4C1 N4C2; do
    P=$OUT/ext_train/arm_$a/checkpoint_r1.pt
    [ -f "$P" ] && CK="$CK ${a}=$P"
  done
  stamp "screening N4b: $CK"
  OUTPUT=$OUT/funnel/screen_m20_N4b STAGE=screen-m20 CHECKPOINTS="$CK" \
    PHYSICAL_GPU=3 PYTHON_BIN=$PY bash scripts/run_expansion_funnel.sh \
    > "$OUT/funnel/screen_m20_N4b.log" 2>&1
  if [ -f "$OUT/funnel/screen_m20_N4b/STAGE_COMPLETE.json" ]; then
    touch "$MK/screen_n4b.done"; stamp "screen N4b done"
  else
    stamp "screen N4b FAILED"; exit 1
  fi
else
  stamp "screen N4b already done"
fi

touch "$MK/gpu3_all.done"
stamp "GPU3 CHAIN COMPLETE"
echo NIGHT4_GPU3_DONE
