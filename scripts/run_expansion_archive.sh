#!/usr/bin/env bash
set -euo pipefail

# SFM2 expansion archive gathering (no update). Override only the output root,
# physical GPU, sampling checkpoint, or round index through the environment
# variables below. The acquisition reference stays the canonical r0.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/data3/research1/sfm_hp100_certified_weighted_500x7_2671a94}"
OUTPUT="${OUTPUT:?set OUTPUT to a new /data3/research1 directory}"
PHYSICAL_GPU="${PHYSICAL_GPU:-3}"
ROUND_INDEX="${ROUND_INDEX:-1}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/checkpoints/hp100_pretrained_r0_258999ae.pt}"
CHECKPOINT_SHA="${CHECKPOINT_SHA:-258999ae8ccee8aec5aab92a6f751221d3c15583ac26e0a7ec8311f13316ec44}"
LINEAGES_PER_GAMMA="${LINEAGES_PER_GAMMA:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONPATH="${ROOT}/source_snapshot/overnight_run_07_12_sfm"

exec "${PYTHON_BIN}" \
  "${ROOT}/source_snapshot/overnight_run_07_12_sfm/sfm_hp100_expansion_archive.py" \
  --checkpoint "${CHECKPOINT}" \
  --expected-checkpoint-sha256 "${CHECKPOINT_SHA}" \
  --reference-checkpoint "${ROOT}/checkpoints/hp100_pretrained_r0_258999ae.pt" \
  --expected-reference-checkpoint-sha256 \
    258999ae8ccee8aec5aab92a6f751221d3c15583ac26e0a7ec8311f13316ec44 \
  --pretrain-dataset-root "${DATA_ROOT}" \
  --expected-pretrain-dataset-manifest-sha256 \
    44f2bfa8afbb2318376ae9e188b1b622f102253a4a91f5c8ca0f9634d5041c94 \
  --output "${OUTPUT}" \
  --device cuda:0 \
  --physical-gpu "${PHYSICAL_GPU}" \
  --scene-profile double_density_velocity_ood \
  --round-index "${ROUND_INDEX}" \
  --gammas 0.1,0.2,0.3,0.4,0.5,0.7,1.0 \
  --lineages-per-gamma "${LINEAGES_PER_GAMMA}" \
  --max-steps 180 \
  --max-attempts 32 \
  --ess-target 0.1 \
  --seed 41 \
  --verifier-workers 32 \
  --status-json "${OUTPUT}/STATUS.json" \
  --heartbeat-seconds 30
