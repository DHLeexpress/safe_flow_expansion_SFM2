#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 CHECKPOINT PREFLIGHT PREFLIGHT_SHA256 RUN_NAME" >&2
  exit 2
fi

CHECKPOINT=$1
PREFLIGHT=$2
PREFLIGHT_SHA256=$3
RUN_NAME=$4
if [[ ! $RUN_NAME =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "RUN_NAME must be one path-safe component" >&2
  exit 2
fi
if [[ ! -f $CHECKPOINT || ! -f $PREFLIGHT ]]; then
  echo "checkpoint and preflight must both exist" >&2
  exit 2
fi

GPU3_UUID=GPU-b5993142-760d-a6fe-9430-3d0e65203b6d
EXPECTED_CHECKPOINT_SHA256=1b5179c935d3eeff8824967d707d64cc9bab273949ee1f0e4f190172bab1b215
EXPECTED_PREFLIGHT_SHA256=d04254396145bb632ffba67222de077067d989b2f57e80e185c2134ad599bc4c
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
if [[ $CUDA_VISIBLE_DEVICES != 3 && $CUDA_VISIBLE_DEVICES != "$GPU3_UUID" ]]; then
  echo "qualification is assigned to physical GPU 3; set CUDA_VISIBLE_DEVICES=3" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES

ROOT=/data3/research1
OUTDIR=$ROOT/$RUN_NAME
LOGDIR=$ROOT/launcher_logs
LOG=$LOGDIR/$RUN_NAME.log
GPU_PROVENANCE=$LOGDIR/$RUN_NAME.gpu.txt
PYTHON_BIN=${PYTHON_BIN:-/home/dohyun/miniforge3/bin/python}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
if [[ $CHECKPOINT_SHA256 != "$EXPECTED_CHECKPOINT_SHA256" ]]; then
  echo "promoted Hp10 checkpoint SHA-256 mismatch: $CHECKPOINT_SHA256" >&2
  exit 2
fi

mkdir -p "$LOGDIR"
if [[ -e $OUTDIR || -e $LOG || -e $GPU_PROVENANCE ]]; then
  echo "refusing an existing output, launcher log, or GPU record" >&2
  exit 2
fi
if [[ $PREFLIGHT_SHA256 != "$EXPECTED_PREFLIGHT_SHA256" ]]; then
  echo "reviewed RBF preflight SHA-256 mismatch: $PREFLIGHT_SHA256" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for the GPU provenance gate" >&2
  exit 2
fi
OBSERVED_GPU_UUID=$(nvidia-smi -i 3 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
if [[ $OBSERVED_GPU_UUID != "$GPU3_UUID" ]]; then
  echo "physical GPU 3 UUID mismatch: $OBSERVED_GPU_UUID" >&2
  exit 2
fi
COMPUTE_ROWS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null || true)
if grep -Fq "$GPU3_UUID" <<<"$COMPUTE_ROWS"; then
  echo "physical GPU 3 already has a compute process; refusing to share it" >&2
  exit 2
fi
nvidia-smi -i 3 \
  --query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader > "$GPU_PROVENANCE"
GPU_PROVENANCE_SHA256=$(sha256sum "$GPU_PROVENANCE" | awk '{print $1}')

"$PYTHON_BIN" "$SCRIPT_DIR/sfm_b1_adaptive_k64_study.py" \
  --checkpoint "$CHECKPOINT" \
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --preflight "$PREFLIGHT" \
  --expected-preflight-sha256 "$PREFLIGHT_SHA256" \
  --gpu-provenance "$GPU_PROVENANCE" \
  --expected-gpu-provenance-sha256 "$GPU_PROVENANCE_SHA256" \
  --scene-profile double_density_velocity_ood \
  --outdir "$OUTDIR" \
  --rounds 5 \
  --device cuda:0 \
  --workers 32 2>&1 | tee "$LOG"
