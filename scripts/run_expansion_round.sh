#!/usr/bin/env bash
set -euo pipefail

# One declared SFM2 expansion recipe: gather -> positive-minus-alpha-negative
# update -> checkpoint, per round. Recipe knobs come from the environment; the
# canonical r0, its digest, and the frozen acquisition contract do not move.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/data3/research1/sfm_hp100_certified_weighted_500x7_2671a94}"
OUTPUT="${OUTPUT:?set OUTPUT to a new /data3/research1 directory}"
RECIPE_ID="${RECIPE_ID:?set RECIPE_ID to the declared recipe id, e.g. E4}"
PHYSICAL_GPU="${PHYSICAL_GPU:-3}"
ALPHA="${ALPHA:-0.0}"
ROUNDS="${ROUNDS:-5}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
EXPOSURE_PASSES="${EXPOSURE_PASSES:-1}"
OPTIMIZER_SCOPE="${OPTIMIZER_SCOPE:-trunk_and_head}"
LINEAGES_PER_GAMMA="${LINEAGES_PER_GAMMA:-8}"
REUSE_ARCHIVE="${REUSE_ARCHIVE:-}"
RESUME_FROM="${RESUME_FROM:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONPATH="${ROOT}/source_snapshot/overnight_run_07_12_sfm"

REUSE_ARGS=()
if [[ -n "${REUSE_ARCHIVE}" ]]; then
  REUSE_ARGS=(--reuse-archive "${REUSE_ARCHIVE}")
fi
if [[ -n "${RESUME_FROM}" ]]; then
  REUSE_ARGS+=(--resume-from "${RESUME_FROM}")
fi

exec "${PYTHON_BIN}" \
  "${ROOT}/source_snapshot/overnight_run_07_12_sfm/sfm_hp100_expansion_round.py" \
  --checkpoint "${ROOT}/checkpoints/hp100_pretrained_r0_258999ae.pt" \
  --expected-checkpoint-sha256 \
    258999ae8ccee8aec5aab92a6f751221d3c15583ac26e0a7ec8311f13316ec44 \
  --pretrain-dataset-root "${DATA_ROOT}" \
  --expected-pretrain-dataset-manifest-sha256 \
    44f2bfa8afbb2318376ae9e188b1b622f102253a4a91f5c8ca0f9634d5041c94 \
  --output "${OUTPUT}" \
  --recipe-id "${RECIPE_ID}" \
  --device cuda:0 \
  --physical-gpu "${PHYSICAL_GPU}" \
  --verifier-workers 32 \
  --rounds "${ROUNDS}" \
  --alpha "${ALPHA}" \
  --learning-rate "${LEARNING_RATE}" \
  --batch-size 64 \
  --exposure-passes "${EXPOSURE_PASSES}" \
  --optimizer-scope "${OPTIMIZER_SCOPE}" \
  --grad-clip-norm 1.0 \
  --max-relative-parameter-drift 0.25 \
  --train-mode eval \
  --update-seed 2 \
  --scene-profile double_density_velocity_ood \
  --gammas 0.1,0.2,0.3,0.4,0.5,0.7,1.0 \
  --lineages-per-gamma "${LINEAGES_PER_GAMMA}" \
  --max-steps 180 \
  --max-attempts 32 \
  --ess-target 0.1 \
  --seed 41 \
  "${REUSE_ARGS[@]}" \
  --status-json "${OUTPUT}/STATUS.json" \
  --heartbeat-seconds 30
