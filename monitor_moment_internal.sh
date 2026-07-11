#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-}"
MODE="${2:-}"
LOG_ROOT="${LOG_ROOT:-./logs/moment_internal_parallel}"

echo "Time: $(date)"
echo ""
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
echo ""

echo "Running MOMENT internal processes:"
ps -ef | grep moment_internal_baselines_multidataset.py | grep -v grep || true
echo ""

pattern="*.log"
if [[ -n "${DATASET}" && -n "${MODE}" ]]; then
  pattern="${DATASET}_${MODE}_*.log"
elif [[ -n "${DATASET}" ]]; then
  pattern="${DATASET}_*.log"
fi

find "${LOG_ROOT}" -type f -name "${pattern}" -print | sort | while read -r f; do
  echo "----------------------------------------------------------------------"
  echo "${f}"
  tail -n 8 "${f}" || true
done
