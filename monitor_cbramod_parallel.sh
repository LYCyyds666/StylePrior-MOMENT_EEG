#!/usr/bin/env bash
set -euo pipefail

# monitor_cbramod_parallel.sh
# Usage:
#   bash monitor_cbramod_parallel.sh
#   bash monitor_cbramod_parallel.sh REFED
#   bash monitor_cbramod_parallel.sh SleepEDF

DATASET="${1:-}"
LOG_ROOT="${LOG_ROOT:-./logs/cbramod_parallel}"

echo "Current time: $(date)"
echo "GPU status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
echo ""

echo "Running CBraMod processes:"
ps -ef | grep cbramod_baseline_multidataset.py | grep -v grep || true
echo ""

if [[ -n "${DATASET}" ]]; then
  echo "Last lines for ${DATASET}:"
  find "${LOG_ROOT}" -type f -name "${DATASET}_*.log" -print | sort | while read -r f; do
    echo "----------------------------------------------------------------------"
    echo "${f}"
    tail -n 8 "${f}" || true
  done
else
  echo "Recent logs:"
  find "${LOG_ROOT}" -type f -name "*.log" -print | sort | tail -n 16 | while read -r f; do
    echo "----------------------------------------------------------------------"
    echo "${f}"
    tail -n 5 "${f}" || true
  done
fi
