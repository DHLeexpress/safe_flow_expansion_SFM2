#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 CHECKPOINT PREFLIGHT PREFLIGHT_SHA256 RUN_NAME [--runtime-gate-only | --runtime-forecast PATH]" >&2
  exit 2
fi

CHECKPOINT=$1
PREFLIGHT=$2
PREFLIGHT_SHA256=$3
RUN_NAME=$4
shift 4

if [[ ! $RUN_NAME =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "RUN_NAME must be one path-safe component" >&2
  exit 2
fi
if [[ $# -eq 1 && $1 != "--runtime-gate-only" ]]; then
  echo "one trailing argument must be --runtime-gate-only" >&2
  exit 2
fi
if [[ $# -eq 2 && $1 != "--runtime-forecast" ]]; then
  echo "two trailing arguments must be --runtime-forecast PATH" >&2
  exit 2
fi
if [[ $# -gt 2 ]]; then
  echo "too many trailing arguments" >&2
  exit 2
fi

ROOT=/data3/research1
OUTDIR=$ROOT/$RUN_NAME
LOGDIR=$ROOT/launcher_logs
LOG=$LOGDIR/$RUN_NAME.log
PYTHON_BIN=${PYTHON_BIN:-/home/dohyun/miniforge3/bin/python}

mkdir -p "$LOGDIR"
if [[ -e $OUTDIR || -e $LOG ]]; then
  echo "refusing an existing output or launcher log: $OUTDIR / $LOG" >&2
  exit 2
fi

"$PYTHON_BIN" overnight_run_07_12_sfm/sfm_b1_alpha_steps_sweep.py \
  --checkpoint "$CHECKPOINT" \
  --preflight "$PREFLIGHT" \
  --expected-preflight-sha256 "$PREFLIGHT_SHA256" \
  --scene-profile double_density_velocity_ood \
  --outdir "$OUTDIR" \
  --rounds 20 \
  --workers 8 \
  --tune-M 10 \
  --screen-M 50 \
  --confirm-M 100 \
  --max-hours 6 \
  "$@" 2>&1 | tee "$LOG"
