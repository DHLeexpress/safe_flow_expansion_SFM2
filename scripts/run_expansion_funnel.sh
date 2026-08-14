#!/usr/bin/env bash
set -euo pipefail

# Staged raw evaluation funnel for SFM2 expansion checkpoints. STAGE selects
# screen-m20 / shortlist-m50 / confirm-m100; CHECKPOINTS is a space-separated
# list of label=path pairs. confirm-m100 additionally needs SEARCH_CONTRACT;
# INCLUDE_ID=1 adds the optional matched-ID screening bank to screen-m20.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${OUTPUT:?set OUTPUT to a new /data3/research1 directory}"
STAGE="${STAGE:?set STAGE to screen-m20, shortlist-m50, or confirm-m100}"
CHECKPOINTS="${CHECKPOINTS:?set CHECKPOINTS to space-separated label=path pairs}"
PHYSICAL_GPU="${PHYSICAL_GPU:-3}"
SEARCH_CONTRACT="${SEARCH_CONTRACT:-}"
INCLUDE_ID="${INCLUDE_ID:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONPATH="${ROOT}/source_snapshot/overnight_run_07_12_sfm"

CHECKPOINT_ARGS=()
for pair in ${CHECKPOINTS}; do
  CHECKPOINT_ARGS+=(--checkpoint "${pair}")
done
CONTRACT_ARGS=()
if [[ -n "${SEARCH_CONTRACT}" ]]; then
  CONTRACT_ARGS=(--search-contract "${SEARCH_CONTRACT}")
fi
if [[ -n "${INCLUDE_ID}" ]]; then
  CONTRACT_ARGS+=(--include-id)
fi

exec "${PYTHON_BIN}" \
  "${ROOT}/source_snapshot/overnight_run_07_12_sfm/sfm_hp100_expansion_funnel.py" \
  "${STAGE}" \
  "${CHECKPOINT_ARGS[@]}" \
  --output "${OUTPUT}" \
  --device cuda \
  --physical-gpu "${PHYSICAL_GPU}" \
  --verifier-workers 32 \
  "${CONTRACT_ARGS[@]}" \
  --status-json "${OUTPUT}_STATUS.json" \
  --heartbeat-seconds 30
