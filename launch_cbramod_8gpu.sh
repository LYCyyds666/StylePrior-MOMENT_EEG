#!/usr/bin/env bash
set -euo pipefail

# launch_cbramod_8gpu.sh
# Usage:
#   bash launch_cbramod_8gpu.sh REFED
#   bash launch_cbramod_8gpu.sh SleepEDF
#   bash launch_cbramod_8gpu.sh both-seq
#   bash launch_cbramod_8gpu.sh both-split
#
# Recommended:
#   nohup bash launch_cbramod_8gpu.sh REFED > logs/launcher_REFED.log 2>&1 &
#   nohup bash launch_cbramod_8gpu.sh SleepEDF > logs/launcher_SleepEDF.log 2>&1 &

TARGET="${1:-}"
MODE="${MODE:-full}"
EPOCHS="${EPOCHS:-30}"
SEED="${SEED:-2025}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-2}"
OUT_ROOT="${OUT_ROOT:-./results_cbramod_parallel}"
LOG_ROOT="${LOG_ROOT:-./logs/cbramod_parallel}"

# Default: all 8 GPUs. Override if needed:
#   GPUS="0 1 2 3" bash launch_cbramod_8gpu.sh REFED
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

fold_count() {
  case "$1" in
    REFED) echo 32 ;;
    SleepEDF) echo 20 ;;
    *) echo "Unknown dataset: $1" >&2; exit 2 ;;
  esac
}

launch_dataset() {
  local dataset="$1"
  local gpu_list="$2"
  local total_folds
  total_folds="$(fold_count "${dataset}")"

  read -r -a gpu_arr <<< "${gpu_list}"
  local n_gpu="${#gpu_arr[@]}"

  echo "======================================================================"
  echo "Launching CBraMod ${dataset} | mode=${MODE} | folds=${total_folds} | GPUs=${gpu_list}"
  echo "epochs=${EPOCHS}, batch_size=${BATCH_SIZE}, num_workers=${NUM_WORKERS}"
  echo "OUT_ROOT=${OUT_ROOT}"
  echo "LOG_ROOT=${LOG_ROOT}"
  echo "======================================================================"

  local pid_file="${LOG_ROOT}/${dataset}_${MODE}_pids.txt"
  : > "${pid_file}"

  for idx in "${!gpu_arr[@]}"; do
    local gpu="${gpu_arr[$idx]}"
    # Balanced integer partition, end is exclusive.
    local start=$(( idx * total_folds / n_gpu ))
    local end=$(( (idx + 1) * total_folds / n_gpu ))

    if [[ "${start}" -ge "${end}" ]]; then
      echo "Skip GPU ${gpu}: empty fold range ${start}:${end}"
      continue
    fi

    local out_dir="${OUT_ROOT}/${dataset}/gpu${gpu}_fold${start}_${end}"
    local log_file="${LOG_ROOT}/${dataset}_${MODE}_gpu${gpu}_fold${start}_${end}.log"
    mkdir -p "${out_dir}"

    echo "[${dataset}] GPU ${gpu}: folds ${start}:${end} -> ${log_file}"

    CUDA_VISIBLE_DEVICES="${gpu}" python cbramod_baseline_multidataset.py \
      --dataset "${dataset}" \
      --mode "${MODE}" \
      --epochs "${EPOCHS}" \
      --seed "${SEED}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --start_fold "${start}" \
      --end_fold "${end}" \
      --output_dir "${out_dir}" \
      > "${log_file}" 2>&1 &

    local pid=$!
    echo "${pid} ${dataset} gpu${gpu} folds_${start}_${end} ${log_file}" | tee -a "${pid_file}"
    sleep 2
  done

  echo "Started jobs for ${dataset}. PID file: ${pid_file}"
  echo "Waiting for ${dataset} jobs..."

  local failed=0
  while read -r pid rest; do
    [[ -z "${pid}" ]] && continue
    if wait "${pid}"; then
      echo "OK: ${pid} ${rest}"
    else
      echo "FAILED: ${pid} ${rest}"
      failed=1
    fi
  done < "${pid_file}"

  echo "All ${dataset} jobs finished. Merging results..."

  python merge_cbramod_parallel.py \
    --dataset "${dataset}" \
    --mode "${MODE}" \
    --seed "${SEED}" \
    --root "${OUT_ROOT}/${dataset}" \
    --expected_folds "${total_folds}" \
    --out_prefix "${OUT_ROOT}/${dataset}/cbramod_${dataset}_${MODE}_merged"

  if [[ "${failed}" -ne 0 ]]; then
    echo "WARNING: one or more ${dataset} jobs failed. Check logs under ${LOG_ROOT}."
    return 1
  fi

  echo "Done ${dataset}."
}

case "${TARGET}" in
  REFED)
    launch_dataset "REFED" "${GPUS}"
    ;;
  SleepEDF)
    launch_dataset "SleepEDF" "${GPUS}"
    ;;
  both-seq)
    launch_dataset "REFED" "${GPUS}"
    launch_dataset "SleepEDF" "${GPUS}"
    ;;
  both-split)
    # Run both datasets at the same time: 4 GPUs each.
    # Use this only if CPU RAM and disk IO are comfortable.
    echo "Launching SleepEDF on GPUs 0-3 and REFED on GPUs 4-7 concurrently."
    (GPUS="0 1 2 3" launch_dataset "SleepEDF" "0 1 2 3") &
    p1=$!
    (GPUS="4 5 6 7" launch_dataset "REFED" "4 5 6 7") &
    p2=$!
    wait "$p1"
    wait "$p2"
    ;;
  *)
    echo "Usage: bash launch_cbramod_8gpu.sh REFED|SleepEDF|both-seq|both-split"
    echo ""
    echo "Recommended:"
    echo "  nohup bash launch_cbramod_8gpu.sh REFED > logs/launcher_REFED.log 2>&1 &"
    echo "  nohup bash launch_cbramod_8gpu.sh SleepEDF > logs/launcher_SleepEDF.log 2>&1 &"
    exit 2
    ;;
esac
